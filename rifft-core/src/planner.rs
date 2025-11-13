use crate::types::{FftDirection, PlanKey, Result, TensorDType};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use rustfft::Fft;
use rustfft::FftPlanner;
use std::collections::{HashMap, VecDeque};
use std::env;
use std::sync::Arc;

#[derive(Clone)]
pub struct PlanEntry {
    pub key: PlanKey,
    pub row_fft: Arc<dyn Fft<f32> + Send + Sync>,
    pub col_fft: Arc<dyn Fft<f32> + Send + Sync>,
    pub len: usize,
}

pub struct Planner {
    cache: RwLock<HashMap<PlanKey, Arc<PlanEntry>>>,
    fifo: RwLock<VecDeque<PlanKey>>,
    planner: RwLock<FftPlanner<f32>>,
}

impl Planner {
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            fifo: RwLock::new(VecDeque::new()),
            planner: RwLock::new(FftPlanner::new()),
        }
    }

    pub fn plan(
        &self,
        height: usize,
        width: usize,
        direction: FftDirection,
        simd_enabled: bool,
    ) -> Result<Arc<PlanEntry>> {
        let key = PlanKey {
            height,
            width,
            direction,
            dtype: TensorDType::Complex32,
            simd_enabled,
        };
        if let Some(entry) = self.cache.read().get(&key).cloned() {
            return Ok(entry);
        }

        let mut planner = self.planner.write();
        let rust_dir = match direction {
            FftDirection::Forward => rustfft::FftDirection::Forward,
            FftDirection::Inverse => rustfft::FftDirection::Inverse,
        };
        let row_fft = planner.plan_fft(width, rust_dir);
        let col_fft = planner.plan_fft(height, rust_dir);
        drop(planner);

        let entry = Arc::new(PlanEntry {
            key,
            row_fft,
            col_fft,
            len: height * width,
        });
        let mut cache = self.cache.write();
        cache.insert(key, entry.clone());

        if should_track_small_plan(height, width) {
            let mut fifo = self.fifo.write();
            fifo.push_back(key);
            let max_small = *SMALL_PLAN_CACHE;
            while fifo.len() > max_small {
                if let Some(old_key) = fifo.pop_front() {
                    cache.remove(&old_key);
                }
            }
        }

        Ok(entry)
    }
}

fn env_usize(var: &str, default: usize) -> usize {
    env::var(var)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

static SMALL_PLAN_LIMIT: Lazy<usize> = Lazy::new(|| env_usize("RUSTFFT_SMALL_MAX", 512));
static SMALL_PLAN_CACHE: Lazy<usize> = Lazy::new(|| env_usize("RUSTFFT_SMALL_CACHE", 32));

fn should_track_small_plan(height: usize, width: usize) -> bool {
    let max_dim = height.max(width);
    max_dim <= *SMALL_PLAN_LIMIT
}

pub static GLOBAL_PLANNER: Lazy<Arc<Planner>> = Lazy::new(|| Arc::new(Planner::new()));
