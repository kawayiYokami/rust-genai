//! A streamed Responses answer must end with a terminal event
//! (`response.completed` / `failed` / `incomplete` / `error`), or `[DONE]` for
//! endpoints that end with that sentinel. An EOF before any of them means the
//! stream was cut: that must surface as an error, not as a finished reply.

use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener};

use futures::StreamExt;
use genai::adapter::AdapterKind;
use genai::chat::{ChatOptions, ChatRequest, ChatStreamEvent};
use genai::resolver::{AuthData, Endpoint};
use genai::{Client, ModelIden, ServiceTarget};

fn serve(body: String) -> u16 {
	let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
	let port = listener.local_addr().expect("addr").port();
	std::thread::spawn(move || {
		let (mut stream, _) = listener.accept().expect("accept");
		let mut buf = [0_u8; 8192];
		let _ = stream.read(&mut buf);
		let header = format!(
			"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
			body.len()
		);
		stream.write_all(header.as_bytes()).expect("header");
		stream.write_all(body.as_bytes()).expect("body");
		let _ = stream.shutdown(Shutdown::Both);
	});
	port
}

/// Raw SSE body from already-serialized `data:` payloads.
fn sse_raw(payloads: &[&str]) -> String {
	let mut body = String::new();
	for payload in payloads {
		body.push_str("data: ");
		body.push_str(payload);
		body.push_str("\n\n");
	}
	body
}

fn sse(events: &[serde_json::Value]) -> String {
	let payloads: Vec<String> = events.iter().map(serde_json::Value::to_string).collect();
	let refs: Vec<&str> = payloads.iter().map(String::as_str).collect();
	sse_raw(&refs)
}

const TEXT_DELTA: &str =
	r#"{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"hi"}"#;

/// Runs one stream and reports whether it ended normally, plus its error
/// rendered as text when it did not.
async fn run(body: String) -> (bool, Option<String>) {
	let port = serve(body);
	let client = Client::builder().build().expect("client");
	let target = ServiceTarget {
		endpoint: Endpoint::from_owned(format!("http://127.0.0.1:{port}/v1/")),
		auth: AuthData::Key("test".to_string()),
		model: ModelIden::new(AdapterKind::OpenAIResp, "gpt-5"),
	};
	let options = ChatOptions::default();
	let response = client
		.exec_chat_stream(target, ChatRequest::from_user("hi"), Some(&options))
		.await
		.expect("stream");

	let mut stream = response.stream;
	let mut saw_end = false;
	let mut error = None;
	while let Some(event) = stream.next().await {
		match event {
			Ok(ChatStreamEvent::End(_)) => saw_end = true,
			Ok(_) => {}
			Err(err) => {
				error = Some(err.to_string());
				break;
			}
		}
	}
	(saw_end, error)
}

#[tokio::test]
async fn stream_ending_without_terminal_event_is_an_error() {
	// Content arrived, then the stream closed without `response.completed`.
	let (saw_end, error) = run(sse_raw(&[TEXT_DELTA])).await;

	assert!(!saw_end, "a cut stream must not end normally");
	let error = error.expect("a cut stream must surface an error");
	assert!(
		error.contains("terminal event"),
		"the error should name the missing terminal event, got: {error}"
	);
}

#[tokio::test]
async fn done_marker_counts_as_terminal_event() {
	// Some compatible endpoints end with `[DONE]` instead of `response.completed`.
	let (saw_end, error) = run(sse_raw(&[TEXT_DELTA, "[DONE]"])).await;

	assert!(error.is_none(), "`[DONE]` must not be treated as a cut stream: {error:?}");
	assert!(saw_end, "`[DONE]` should produce a normal end");
}

#[tokio::test]
async fn unparsable_terminal_event_is_reported_as_that_not_as_a_cut_stream() {
	// The terminal frame arrived but failed to deserialize (missing `model`).
	// That is not a cut stream, and the message must not say it is one.
	let completed_without_model =
		r#"{"type":"response.completed","response":{"id":"resp1","status":"completed"}}"#;
	let (saw_end, error) = run(sse_raw(&[TEXT_DELTA, completed_without_model])).await;

	assert!(!saw_end, "an unparsable terminal event must not end normally");
	let error = error.expect("must surface an error");
	assert!(
		error.contains("could not be parsed"),
		"the error should blame the terminal event's parse, got: {error}"
	);
	assert!(
		error.contains("response.completed"),
		"the error should name the frame kind, got: {error}"
	);
}

#[tokio::test]
async fn completed_response_ends_normally() {
	let events = vec![
		serde_json::json!({ "type": "response.created", "response": { "id": "resp1", "status": "in_progress", "model": "gpt-5" } }),
		serde_json::json!({
			"type": "response.output_text.delta",
			"output_index": 0,
			"content_index": 0,
			"delta": "hi"
		}),
		serde_json::json!({ "type": "response.completed", "response": { "id": "resp1", "status": "completed", "model": "gpt-5" } }),
	];
	let (saw_end, error) = run(sse(&events)).await;

	assert!(error.is_none(), "a completed stream must not error: {error:?}");
	assert!(saw_end, "a completed stream should end normally");
}
