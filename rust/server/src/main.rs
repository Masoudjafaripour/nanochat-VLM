//! `nanochat-vlm-server`: REST API over the Rust inference engine.
//!
//! Endpoints
//! * `GET  /health`        liveness
//! * `GET  /v1/info`       model configuration
//! * `POST /v1/generate`   `{prompt, image_base64?, max_new_tokens?, temperature?, top_k?, top_p?,
//!                           seed?, raw?, stream?}`
//!   - `stream: false` -> `{text, tokens, stats}`
//!   - `stream: true`  -> Server-Sent Events: `token` events with text deltas, then one `done`
//!     event with the stats.
//!
//! Concurrency model: the engine is shared read-only (`Arc`); every request gets its own
//! KV cache and runs on Tokio's blocking pool, so requests decode in parallel. A semaphore
//! caps concurrent generations (`--max-concurrency`); waiting requests queue in FIFO
//! order, and the wait is reported as `queue_ms`. On CPU, candle's matmuls are already
//! multi-threaded, so more concurrency raises total throughput only up to a point.
//! The benchmarks measure where that point is instead of guessing.

use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine as _;
use candle_core::{DType, Device};
use clap::Parser;
use futures::stream::Stream;
use serde::Deserialize;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, Semaphore};
use tokio_stream::wrappers::ReceiverStream;
use vlm_inference::tokenizer::StreamDecoder;
use vlm_inference::{Engine, GenerateOptions, GenerationStats, SamplingParams, Segment};

#[derive(Parser)]
#[command(name = "nanochat-vlm-server", about = "nanochat-VLM REST API")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "127.0.0.1:8080")]
    addr: SocketAddr,
    #[arg(long, default_value_t = 4)]
    max_concurrency: usize,
    /// Upper bound on max_new_tokens a client may request.
    #[arg(long, default_value_t = 512)]
    max_tokens_limit: usize,
    #[arg(long)]
    cuda: bool,
}

struct AppState {
    engine: Engine,
    permits: Arc<Semaphore>,
    max_tokens_limit: usize,
}

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
    #[serde(default)]
    image_base64: Option<String>,
    #[serde(default = "default_max_new")]
    max_new_tokens: usize,
    #[serde(default)]
    temperature: f32,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default = "default_seed")]
    seed: u64,
    #[serde(default)]
    raw: bool,
    #[serde(default)]
    stream: bool,
}

fn default_max_new() -> usize {
    64
}
fn default_seed() -> u64 {
    42
}

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({ "error": self.1 }))).into_response()
    }
}

fn bad_request(e: impl std::fmt::Display) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, e.to_string())
}

fn internal(e: impl std::fmt::Display) -> ApiError {
    ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

/// Decode + preprocess the image and build prompt segments. CPU-bound, so callers run
/// it on the blocking pool.
fn build_prompt(engine: &Engine, req: &GenerateRequest) -> anyhow::Result<(Vec<Segment>, Vec<u32>, f64)> {
    let t = Instant::now();
    let patches = match &req.image_base64 {
        Some(b64) => {
            let bytes = base64::engine::general_purpose::STANDARD.decode(b64.trim())?;
            Some(vlm_preprocessing::preprocess_bytes(&bytes, &engine.preprocess)?)
        }
        None => None,
    };
    let preprocess_ms = t.elapsed().as_secs_f64() * 1e3;
    let tok = engine.tokenizer()?;
    if req.raw {
        let mut segs = vec![Segment::Tokens(vec![tok.bos_id()])];
        if let Some(p) = patches {
            segs.push(Segment::Image(p));
        }
        segs.push(Segment::Tokens(tok.encode(&req.prompt)));
        Ok((segs, vec![tok.bos_id()], preprocess_ms))
    } else {
        let (segs, stop) = engine.chat_prompt(&req.prompt, patches)?;
        Ok((segs, stop, preprocess_ms))
    }
}

fn opts_of(req: &GenerateRequest, stop: Vec<u32>) -> GenerateOptions {
    GenerateOptions {
        max_new_tokens: req.max_new_tokens,
        sampling: SamplingParams { temperature: req.temperature, top_k: req.top_k, top_p: req.top_p, seed: req.seed },
        stop_tokens: stop,
        ..Default::default()
    }
}

fn stats_json(st: &GenerationStats, queue_ms: f64, preprocess_ms: f64) -> serde_json::Value {
    let mut v = serde_json::to_value(st).unwrap_or_default();
    v["queue_ms"] = queue_ms.into();
    v["preprocess_ms"] = preprocess_ms.into();
    v
}

async fn generate(State(app): State<Arc<AppState>>, Json(req): Json<GenerateRequest>) -> Result<Response, ApiError> {
    if req.max_new_tokens == 0 || req.max_new_tokens > app.max_tokens_limit {
        return Err(bad_request(format!("max_new_tokens must be in 1..={}", app.max_tokens_limit)));
    }
    let queued = Instant::now();
    let permit = app.permits.clone().acquire_owned().await.map_err(internal)?;
    let queue_ms = queued.elapsed().as_secs_f64() * 1e3;

    if !req.stream {
        let app2 = app.clone();
        let res = tokio::task::spawn_blocking(move || -> anyhow::Result<serde_json::Value> {
            let _permit = permit;
            let (segs, stop, pre_ms) = build_prompt(&app2.engine, &req)?;
            let opts = opts_of(&req, stop.clone());
            let (toks, st) = app2.engine.generate(&segs, &opts, |_| true)?;
            let text_ids: Vec<u32> = toks.iter().copied().filter(|t| !stop.contains(t)).collect();
            let text = app2.engine.tokenizer()?.decode(&text_ids);
            Ok(serde_json::json!({ "text": text, "tokens": toks, "stats": stats_json(&st, queue_ms, pre_ms) }))
        })
        .await
        .map_err(internal)?;
        return match res {
            Ok(v) => Ok(Json(v).into_response()),
            Err(e) => Err(bad_request(e)),
        };
    }

    // Streaming: the blocking task pushes events into a channel the SSE response drains.
    let (tx, rx) = mpsc::channel::<Event>(64);
    let app2 = app.clone();
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let run = || -> anyhow::Result<serde_json::Value> {
            let (segs, stop, pre_ms) = build_prompt(&app2.engine, &req)?;
            let opts = opts_of(&req, stop.clone());
            let tok = app2.engine.tokenizer()?;
            let mut dec = StreamDecoder::default();
            let (_, st) = app2.engine.generate(&segs, &opts, |id| {
                if stop.contains(&id) {
                    return true;
                }
                let delta = dec.push(tok, id);
                let ev = Event::default().event("token").data(serde_json::json!({"id": id, "text": delta}).to_string());
                // Client went away -> stop generating.
                tx.blocking_send(ev).is_ok()
            })?;
            Ok(stats_json(&st, queue_ms, pre_ms))
        };
        let ev = match run() {
            Ok(stats) => Event::default().event("done").data(stats.to_string()),
            Err(e) => Event::default().event("error").data(e.to_string()),
        };
        let _ = tx.blocking_send(ev);
    });
    let stream = ReceiverStream::new(rx);
    Ok(sse(stream).into_response())
}

fn sse(s: ReceiverStream<Event>) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    use futures::StreamExt;
    Sse::new(s.map(Ok))
}

async fn info(State(app): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::to_value(&app.engine.model.cfg).unwrap_or_default())
}

fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/info", get(info))
        .route("/v1/generate", post(generate))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(state)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = if args.cuda { Device::new_cuda(0)? } else { Device::Cpu };
    let engine = Engine::load(&args.model, &device, DType::F32)?;
    let state = Arc::new(AppState {
        engine,
        permits: Arc::new(Semaphore::new(args.max_concurrency.max(1))),
        max_tokens_limit: args.max_tokens_limit,
    });
    let listener = tokio::net::TcpListener::bind(args.addr).await?;
    eprintln!("nanochat-vlm-server listening on http://{} (max_concurrency={})", args.addr, args.max_concurrency);
    axum::serve(listener, router(state)).await?;
    Ok(())
}
