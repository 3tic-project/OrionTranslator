use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone)]
pub struct NerClient {
    client: Client,
    base_url: String,
}

#[derive(Debug, Serialize)]
struct NerRequest {
    texts: Vec<String>,
    combine_entities: bool,
    filter_types: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct NerResponse {
    pub entities: Vec<Vec<EntityOutput>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EntityOutput {
    pub entity_type: String,
    pub word: String,
    pub score: f32,
    pub start: usize,
    pub end: usize,
}

impl NerClient {
    pub fn new(base_url: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    pub async fn health_check(&self) -> Result<bool> {
        let resp = self
            .client
            .get(format!("{}/health", self.base_url))
            .timeout(Duration::from_secs(5))
            .send()
            .await?;
        Ok(resp.status().is_success())
    }

    /// Call NER API with a batch of texts, returns entities per text
    pub async fn predict_batch(
        &self,
        texts: Vec<String>,
        filter_types: Vec<String>,
    ) -> Result<Vec<Vec<EntityOutput>>> {
        let request = NerRequest {
            texts,
            combine_entities: true,
            filter_types,
        };

        let resp = self
            .client
            .post(format!("{}/ner", self.base_url))
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("NER API returned {}: {}", status, body);
        }

        let result: NerResponse = resp.json().await?;
        Ok(result.entities)
    }
}
