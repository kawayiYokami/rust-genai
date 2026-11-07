# Request inspection (`RequestTap`)

genai normalizes requests per provider and then sends them. `capture_raw_body` gives you the
response, `ChatFrameSink` gives you the returned frames — `RequestTap` gives you the request
that went out, in the shape it went out.

The library **only hands it over**. It never writes it to disk, never redacts credentials,
never logs it, and keeps no copy. Whatever a tap does with the snapshot is the caller's business.

## The three request shapes

| Call | `RequestCtx.kind` |
| --- | --- |
| `exec_chat` | `RequestKind::Chat` |
| `exec_chat_stream` | `RequestKind::ChatStream` |
| `exec_embed` / `embed` / `embed_batch` | `RequestKind::Embed` |

`all_model_names` is not a chat or embed request and never fires a tap.

## Minimal use

```rust
use genai::chat::{CollectorRequestTap, RequestTap};
use genai::{Client, chat::ChatRequest};
use std::sync::Arc;

let collector = Arc::new(CollectorRequestTap::new());
let client = Client::builder()
    .with_request_tap_arc(collector.clone())   // every call on this client
    .build()?;

let chat_req = ChatRequest::from_user("hello");
let _res = client.exec_chat("gpt-5.4-mini", chat_req, None).await?;

if let Some(captured) = collector.take() {
    println!("[{}] {}", captured.ctx.kind, captured.request.url);
    println!("{}", captured.request.payload);
    for (name, value) in captured.request.headers.iter() {
        println!("{name}: {value}");
    }
}
```

Scope it to one call instead of the whole client with
`ChatOptions::with_request_tap` / `with_request_tap_arc`. Like `raw_frame_sink`, a per-call tap
**replaces** the client-level one for that call rather than running alongside it.

`CollectorRequestTap` is the built-in implementation and keeps only the most recent request;
use `take()` to move it out. A client-level tap sees concurrent calls through the same `&self`,
so a tap that needs to keep more than "the last one" should bucket by `RequestCtx` itself.

## What "final" means

The tap fires after the adapter has shaped the request and after every header merge and auth
override: the `url` is the adapter-expanded endpoint (or the `AuthData::RequestOverride` url),
`headers` already carry the provider auth header plus client-level and per-call extra headers,
and `payload` is the provider-specific body, including flags like `stream`.

Two things are deliberately outside the snapshot:

- Transport headers added by `reqwest` itself (`host`, `content-length`, `accept`, `user-agent`,
  TLS-level details). They are not visible to the library.
- The response. Use `capture_raw_body` (non-streaming) or `ChatFrameSink` (streaming) for that.

## Behavior worth knowing

- Fires once per call, synchronously, on the caller's task. A slow tap delays its request; do not
  block inside one. `CollectorRequestTap` never blocks — use a channel or queue of your own for
  anything else.
- The snapshot borrows the caller's url/headers/payload. Copy what you keep
  (`RequestRef::to_owned_request`) before returning.
- A panicking tap panics the call. The library does not catch it.
- Nothing is redacted: `authorization` is in the snapshot as sent.
