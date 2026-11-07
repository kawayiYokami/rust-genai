//! Request snapshots, handed to a [`RequestTap`] right before the request goes out.
//!
//! The library never records what it sends: `capture_raw_body` covers the response,
//! `ChatFrameSink` covers the returned frames, and the request itself is dropped once it
//! reaches the transport. A [`RequestTap`] closes that gap — it receives the **final** url,
//! headers and payload (adapter-shaped, every header merge and auth override already applied)
//! of each chat, chat-stream and embed call.
//!
//! The library only hands the snapshot over. It never writes it anywhere, never redacts it,
//! never logs it, and keeps no copy: whatever a tap does with it — diff, dump, doctor — is
//! entirely the caller's business. A tap that panics panics the call; keep tap bodies cheap
//! and never do blocking I/O in one (see [`CollectorRequestTap`] for the built-in in-memory one).
//!
//! One tap fires once per call, before the transport:
//!
//! ```
//! use genai::chat::{CollectorRequestTap, RequestTap};
//! use genai::Client;
//! use std::sync::Arc;
//!
//! let collector = Arc::new(CollectorRequestTap::new());
//! let client = Client::builder()
//!     .with_request_tap_arc(collector.clone())
//!     .build()
//!     .expect("client");
//!
//! // ... run a call ...
//!
//! if let Some(captured) = collector.take() {
//!     println!("{} {}", captured.ctx.kind, captured.request.url);
//!     println!("{}", captured.request.payload);
//! }
//! ```
//!
//! Grouping is the tap's own job: a client-level tap sees every call through the same `&self`,
//! including concurrent ones, so bucket by [`RequestCtx`] rather than assume one request at a time.

use crate::Headers;
use crate::ModelIden;
use serde_json::Value;
use std::sync::Mutex;

// region:    --- RequestKind

/// Which of the three request shapes a snapshot belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestKind {
	/// A non-streaming chat call (`exec_chat`).
	Chat,
	/// A streaming chat call (`exec_chat_stream`).
	ChatStream,
	/// A non-streaming embed call (`exec_embed`, `embed`, `embed_batch`).
	Embed,
}

impl std::fmt::Display for RequestKind {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let label = match self {
			RequestKind::Chat => "chat",
			RequestKind::ChatStream => "chat_stream",
			RequestKind::Embed => "embed",
		};
		f.write_str(label)
	}
}

// endregion: --- RequestKind

// region:    --- RequestCtx

/// Identifies the call a request belongs to.
#[derive(Debug, Clone)]
pub struct RequestCtx {
	/// The request shape.
	pub kind: RequestKind,

	/// The resolved model for this call.
	pub model_iden: ModelIden,
}

impl RequestCtx {
	pub(crate) fn new(kind: RequestKind, model_iden: ModelIden) -> Self {
		Self { kind, model_iden }
	}
}

// endregion: --- RequestCtx

// region:    --- RequestRef

/// A borrowed view of one request, exactly as it is about to go on the wire.
#[derive(Debug, Clone, Copy)]
pub struct RequestRef<'a> {
	/// The final url, after adapter shaping and any auth override.
	pub url: &'a str,

	/// The final headers, after client-level and per-call merges and any auth override.
	pub headers: &'a Headers,

	/// The final payload the adapter built.
	pub payload: &'a Value,
}

impl RequestRef<'_> {
	/// Materializes an owned [`RequestSnapshot`], deep-copying all three parts.
	pub fn to_owned_request(&self) -> RequestSnapshot {
		RequestSnapshot {
			url: self.url.to_string(),
			headers: self.headers.clone(),
			payload: self.payload.clone(),
		}
	}
}

// endregion: --- RequestRef

// region:    --- RequestSnapshot

/// An owned copy of one request.
#[derive(Debug, Clone)]
pub struct RequestSnapshot {
	/// The final url.
	pub url: String,

	/// The final headers.
	pub headers: Headers,

	/// The final payload.
	pub payload: Value,
}

// endregion: --- RequestSnapshot

// region:    --- RequestTap

/// Receives the final request of every call, right before it is sent.
///
/// - Fires once per call, synchronously, on the caller's task — a slow tap delays the request.
/// - Concurrent calls share one `&self` when the tap is set on the client; bucket by
///   [`RequestCtx`] when that matters.
/// - Never called for `all_model_names`, which is not a chat or embed request.
/// - The snapshot borrows the caller's url, headers and payload: copy what you keep
///   ([`RequestRef::to_owned_request`]) before returning.
pub trait RequestTap: Send + Sync {
	/// Called once per call, with the request as it is about to be sent.
	fn on_request(&self, ctx: &RequestCtx, request: RequestRef<'_>);
}

impl std::fmt::Debug for dyn RequestTap {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "RequestTap")
	}
}

// endregion: --- RequestTap

// region:    --- Provided Taps

/// Keeps the most recent request in memory.
///
/// "Most recent" is deliberate: a client-level tap sees every call, and the usual question is
/// "what did the call I just made send". Use [`CollectorRequestTap::take`] to move the snapshot
/// out, so a long-lived collector does not keep it alive.
///
/// Concurrent calls race for the slot; the last one to reach the tap wins.
#[derive(Debug, Default)]
pub struct CollectorRequestTap {
	last: Mutex<Option<CapturedRequest>>,
}

impl CollectorRequestTap {
	/// Creates an empty collector.
	pub fn new() -> Self {
		Self::default()
	}

	/// True when no request has been captured yet.
	pub fn is_empty(&self) -> bool {
		self.last.lock().map(|last| last.is_none()).unwrap_or(true)
	}

	/// Clones out the most recent request, if any.
	pub fn last(&self) -> Option<CapturedRequest> {
		self.last.lock().ok().and_then(|last| last.clone())
	}

	/// Takes the most recent request, leaving the collector empty.
	pub fn take(&self) -> Option<CapturedRequest> {
		self.last.lock().ok().and_then(|mut last| last.take())
	}
}

impl RequestTap for CollectorRequestTap {
	fn on_request(&self, ctx: &RequestCtx, request: RequestRef<'_>) {
		if let Ok(mut last) = self.last.lock() {
			*last = Some(CapturedRequest {
				ctx: ctx.clone(),
				request: request.to_owned_request(),
			});
		}
	}
}

/// One request a [`CollectorRequestTap`] kept: the call it belonged to, plus the snapshot.
#[derive(Debug, Clone)]
pub struct CapturedRequest {
	/// Identifies the call.
	pub ctx: RequestCtx,

	/// The request as it was sent.
	pub request: RequestSnapshot,
}

// endregion: --- Provided Taps

// region:    --- Tests

#[cfg(test)]
mod tests {
	type Result<T> = core::result::Result<T, Box<dyn std::error::Error>>;

	use super::*;
	use crate::adapter::AdapterKind;

	fn ctx(kind: RequestKind) -> RequestCtx {
		RequestCtx::new(kind, ModelIden::new(AdapterKind::OpenAI, "gpt-test"))
	}

	fn request_ref<'a>(url: &'a str, headers: &'a Headers, payload: &'a Value) -> RequestRef<'a> {
		RequestRef {
			url,
			headers,
			payload,
		}
	}

	#[test]
	fn test_request_tap_collector_keeps_the_last_request() -> Result<()> {
		// -- Setup & Fixtures
		let tap = CollectorRequestTap::new();
		let headers = Headers::from(("authorization", "Bearer sk-test"));
		let first = serde_json::json!({"n": 1});
		let second = serde_json::json!({"n": 2});

		// -- Exec
		tap.on_request(&ctx(RequestKind::Chat), request_ref("https://a/1", &headers, &first));
		tap.on_request(
			&ctx(RequestKind::ChatStream),
			request_ref("https://a/2", &headers, &second),
		);

		// -- Check
		assert!(!tap.is_empty());
		let captured = tap.take().expect("should have captured a request");
		assert_eq!(captured.request.url, "https://a/2", "the last request wins");
		assert_eq!(captured.ctx.kind, RequestKind::ChatStream);
		assert_eq!(captured.request.payload, second);

		Ok(())
	}

	#[test]
	fn test_request_tap_collector_take_empties_the_slot() -> Result<()> {
		// -- Setup & Fixtures
		let tap = CollectorRequestTap::new();
		assert!(tap.is_empty(), "a fresh collector starts empty");
		assert!(tap.take().is_none());

		let headers = Headers::from(("x-a", "1"));
		let payload = serde_json::json!({"messages": []});

		// -- Exec
		tap.on_request(&ctx(RequestKind::Embed), request_ref("https://a/e", &headers, &payload));
		let taken = tap.take();
		let taken_again = tap.take();

		// -- Check
		assert!(taken.is_some());
		assert!(taken_again.is_none(), "take should leave the collector empty");
		assert!(tap.is_empty());

		Ok(())
	}

	#[test]
	fn test_request_tap_snapshot_copies_every_part() -> Result<()> {
		// -- Setup & Fixtures
		let headers = Headers::from([("authorization".to_string(), "Bearer sk-test".to_string())]);
		let payload = serde_json::json!({"model": "gpt-test"});
		let borrowed = request_ref("https://a/v1/chat", &headers, &payload);

		// -- Exec
		let snapshot = borrowed.to_owned_request();

		// -- Check
		assert_eq!(snapshot.url, "https://a/v1/chat");
		assert_eq!(
			snapshot.headers.iter().find(|(k, _)| *k == "authorization").map(|(_, v)| v.as_str()),
			Some("Bearer sk-test"),
			"headers should be deep-copied"
		);
		assert_eq!(snapshot.payload, payload, "payload should be deep-copied");

		Ok(())
	}

	#[test]
	fn test_request_tap_kind_display() -> Result<()> {
		// -- Exec & Check
		assert_eq!(RequestKind::Chat.to_string(), "chat");
		assert_eq!(RequestKind::ChatStream.to_string(), "chat_stream");
		assert_eq!(RequestKind::Embed.to_string(), "embed");

		Ok(())
	}
}

// endregion: --- Tests
