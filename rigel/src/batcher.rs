use crate::ner::{NerPipeline, NerResult};
use burn::tensor::backend::Backend;
use log::debug;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::time::{timeout, Duration};

/// Batch request
pub struct BatchRequest {
    pub text: String,
    pub response_tx: oneshot::Sender<anyhow::Result<NerResult>>,
}

/// Dynamic batcher configuration
pub struct BatcherConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum wait time (milliseconds)
    pub max_wait_ms: u64,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_wait_ms: 10,
        }
    }
}

/// Dynamic batcher
pub struct DynamicBatcher<B: Backend> {
    request_tx: mpsc::Sender<BatchRequest>,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend + 'static> DynamicBatcher<B> {
    /// Create a new batcher and start background processing task
    pub fn new(pipeline: Arc<Mutex<NerPipeline<B>>>, config: BatcherConfig) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<BatchRequest>(1000);

        tokio::spawn(Self::batch_processor(pipeline, request_rx, config));

        Self {
            request_tx,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Submit a single inference request
    pub async fn predict(&self, text: String) -> anyhow::Result<NerResult> {
        let (response_tx, response_rx) = oneshot::channel();

        let request = BatchRequest { text, response_tx };

        self.request_tx
            .send(request)
            .await
            .map_err(|_| anyhow::anyhow!("Batcher channel closed"))?;

        response_rx
            .await
            .map_err(|_| anyhow::anyhow!("Response channel closed"))?
    }

    /// Background batch processor task
    async fn batch_processor(
        pipeline: Arc<Mutex<NerPipeline<B>>>,
        mut request_rx: mpsc::Receiver<BatchRequest>,
        config: BatcherConfig,
    ) {
        loop {
            let first_request = match request_rx.recv().await {
                Some(req) => req,
                None => break,
            };

            let mut batch: Vec<BatchRequest> = vec![first_request];
            let batch_start = std::time::Instant::now();

            // Collect more requests within time window
            while batch.len() < config.max_batch_size {
                let remaining = Duration::from_millis(config.max_wait_ms)
                    .saturating_sub(batch_start.elapsed());

                if remaining.is_zero() {
                    break;
                }

                match timeout(remaining, request_rx.recv()).await {
                    Ok(Some(req)) => batch.push(req),
                    Ok(None) => break,
                    Err(_) => break,
                }
            }

            let batch_size = batch.len();
            let process_start = std::time::Instant::now();

            {
                let pipeline = pipeline.lock().await;
                let texts: Vec<&str> = batch.iter().map(|r| r.text.as_str()).collect();

                match pipeline.predict_batch(&texts) {
                    Ok(results) => {
                        for (request, result) in batch.into_iter().zip(results.into_iter()) {
                            let _ = request.response_tx.send(Ok(result));
                        }
                    }
                    Err(e) => {
                        let err_msg = e.to_string();
                        for request in batch {
                            let _ = request
                                .response_tx
                                .send(Err(anyhow::anyhow!("{}", err_msg)));
                        }
                    }
                }
            }

            debug!(
                "[DynBatch] size={} | wait: {:?} | process: {:?}",
                batch_size,
                batch_start.elapsed() - process_start.elapsed(),
                process_start.elapsed()
            );
        }
    }
}
