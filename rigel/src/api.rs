use crate::batcher::DynamicBatcher;
use crate::ner::{NerEntity, NerPipeline};
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use burn::tensor::backend::Backend;
use log::debug;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};

pub struct AppState<B: Backend> {
    pub pipeline: Option<Arc<Mutex<NerPipeline<B>>>>,
    pub batcher: Option<Arc<DynamicBatcher<B>>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum TextInput {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Deserialize)]
pub struct NerRequest {
    #[serde(alias = "text")]
    pub texts: TextInput,
    #[serde(default)]
    pub combine_entities: bool,
    pub filter_types: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct EntityOutput {
    pub entity_type: String,
    pub word: String,
    pub score: f32,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
}

pub fn create_router<B: Backend + 'static>(state: Arc<AppState<B>>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(health_check))
        .route("/ner", post(ner_predict::<B>))
        .layer(cors)
        .with_state(state)
}

async fn health_check() -> impl IntoResponse {
    Json(HealthResponse {
        status: "healthy".to_string(),
        model: "bert-japanese-ner-burn-wgpu".to_string(),
    })
}

async fn ner_predict<B: Backend + 'static>(
    State(state): State<Arc<AppState<B>>>,
    Json(request): Json<NerRequest>,
) -> impl IntoResponse {
    let req_start = std::time::Instant::now();
    let texts: Vec<String> = match request.texts {
        TextInput::Single(text) => vec![text],
        TextInput::Multiple(texts) => texts,
    };
    debug!(
        "[API] Received {} text(s), total chars: {}",
        texts.len(),
        texts.iter().map(|t| t.len()).sum::<usize>()
    );

    let results = if let Some(ref batcher) = state.batcher {
        // Dynamic batching mode
        let futures: Vec<_> = texts
            .iter()
            .map(|text| batcher.predict(text.clone()))
            .collect();
        let results: Vec<_> = futures::future::join_all(futures).await;
        let mut ner_results = Vec::new();
        for result in results {
            match result {
                Ok(r) => ner_results.push(r),
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": e.to_string()})),
                    );
                }
            }
        }
        ner_results
    } else if let Some(ref pipeline) = state.pipeline {
        // Direct batch mode (mutex-protected)
        let pipeline = pipeline.lock().await;
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        match pipeline.predict_batch(&text_refs) {
            Ok(r) => r,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                );
            }
        }
    } else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "No inference backend configured"})),
        );
    };

    let mut all_entities: Vec<Vec<EntityOutput>> = Vec::new();
    for result in results {
        let mut entities = result.entities;
        if request.combine_entities {
            entities = combine_adjacent_entities(entities);
        }
        if let Some(ref filter) = request.filter_types {
            entities = filter_entities(entities, filter);
        }
        let output: Vec<EntityOutput> = entities
            .into_iter()
            .map(|e| EntityOutput {
                entity_type: e.label,
                word: e.text,
                score: e.score,
                start: e.start,
                end: e.end,
            })
            .collect();
        all_entities.push(output);
    }

    debug!("[API] Total request time: {:?}", req_start.elapsed());
    (
        StatusCode::OK,
        Json(serde_json::json!({"entities": all_entities})),
    )
}

fn combine_adjacent_entities(entities: Vec<NerEntity>) -> Vec<NerEntity> {
    if entities.is_empty() {
        return entities;
    }

    let mut combined = Vec::new();
    let mut current: Option<NerEntity> = None;

    for entity in entities {
        if let Some(ref mut curr) = current {
            if curr.label == entity.label && curr.end == entity.start {
                curr.text.push_str(&entity.text);
                curr.end = entity.end;
                curr.score = (curr.score + entity.score) / 2.0;
            } else {
                combined.push(current.take().unwrap());
                current = Some(entity);
            }
        } else {
            current = Some(entity);
        }
    }

    if let Some(curr) = current {
        combined.push(curr);
    }

    combined
}

fn filter_entities(entities: Vec<NerEntity>, filter_types: &[String]) -> Vec<NerEntity> {
    entities
        .into_iter()
        .filter(|e| filter_types.iter().any(|t| e.label.contains(t)))
        .collect()
}
