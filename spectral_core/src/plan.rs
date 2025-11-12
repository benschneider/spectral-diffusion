use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use lru::LruCache;
use once_cell::sync::Lazy;
use rustfft::{Fft, FftPlanner};

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum PlanDirection {
    Forward,
    Inverse,
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum PlanDType {
    F32,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
struct PlanKey {
    dtype: PlanDType,
    height: usize,
    width: usize,
    direction: PlanDirection,
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct Plan2D {
    pub height: usize,
    pub width: usize,
    pub direction: PlanDirection,
    pub row_plan: Arc<dyn Fft<f32> + Send + Sync>,
    pub col_plan: Arc<dyn Fft<f32> + Send + Sync>,
}

pub struct PlanCache {
    planner: Mutex<FftPlanner<f32>>,
    cache: Mutex<LruCache<PlanKey, Arc<Plan2D>>>,
}

impl PlanCache {
    pub fn global() -> &'static PlanCache {
        static CACHE: Lazy<PlanCache> = Lazy::new(|| PlanCache::new(64));
        &CACHE
    }

    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).unwrap();
        Self {
            planner: Mutex::new(FftPlanner::new()),
            cache: Mutex::new(LruCache::new(cap)),
        }
    }

    pub fn get_or_build(
        &self,
        height: usize,
        width: usize,
        direction: PlanDirection,
    ) -> (Arc<Plan2D>, Duration) {
        let key = PlanKey {
            dtype: PlanDType::F32,
            height,
            width,
            direction,
        };

        if let Some(existing) = self.cache.lock().unwrap().get(&key) {
            return (existing.clone(), Duration::ZERO);
        }

        let start = Instant::now();
        let mut planner = self.planner.lock().unwrap();
        let row_plan = match direction {
            PlanDirection::Forward => planner.plan_fft_forward(width),
            PlanDirection::Inverse => planner.plan_fft_inverse(width),
        };
        let col_plan = match direction {
            PlanDirection::Forward => planner.plan_fft_forward(height),
            PlanDirection::Inverse => planner.plan_fft_inverse(height),
        };
        drop(planner);
        let elapsed = start.elapsed();

        let plan = Arc::new(Plan2D {
            height,
            width,
            direction,
            row_plan,
            col_plan,
        });

        let mut cache = self.cache.lock().unwrap();
        cache.put(key, plan.clone());

        (plan, elapsed)
    }
}
