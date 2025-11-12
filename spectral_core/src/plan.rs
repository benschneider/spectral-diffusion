use std::collections::HashMap;
use std::env;
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

pub struct SmallPlanCache {
    enabled: bool,
    max_dim: usize,
    plans: HashMap<PlanKey, Arc<Plan2D>>,
}

impl SmallPlanCache {
    pub fn global() -> &'static SmallPlanCache {
        static CACHE: Lazy<SmallPlanCache> = Lazy::new(SmallPlanCache::new);
        &CACHE
    }

    fn new() -> Self {
        let enabled = env::var("RUSTFFT_SMALL_PLANS")
            .map(|v| !matches!(v.as_str(), "0" | "false" | "False"))
            .unwrap_or(true);

        if !enabled {
            return Self {
                enabled: false,
                max_dim: 0,
                plans: HashMap::new(),
            };
        }

        let max_dim = env::var("RUSTFFT_SMALL_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(256);

        let mut sizes = parse_small_sizes(max_dim);
        if sizes.is_empty() {
            sizes.push(64);
        }

        let mut planner = FftPlanner::<f32>::new();
        let mut plans = HashMap::new();

        for &height in &sizes {
            for &width in &sizes {
                for &direction in &[PlanDirection::Forward, PlanDirection::Inverse] {
                    let row_plan = match direction {
                        PlanDirection::Forward => planner.plan_fft_forward(width),
                        PlanDirection::Inverse => planner.plan_fft_inverse(width),
                    };
                    let col_plan = match direction {
                        PlanDirection::Forward => planner.plan_fft_forward(height),
                        PlanDirection::Inverse => planner.plan_fft_inverse(height),
                    };
                    let plan = Arc::new(Plan2D {
                        height,
                        width,
                        direction,
                        row_plan,
                        col_plan,
                    });
                    let key = PlanKey {
                        dtype: PlanDType::F32,
                        height,
                        width,
                        direction,
                    };
                    plans.insert(key, plan);
                }
            }
        }

        Self {
            enabled: true,
            max_dim,
            plans,
        }
    }

    pub fn get(
        &self,
        height: usize,
        width: usize,
        direction: PlanDirection,
    ) -> Option<Arc<Plan2D>> {
        if !self.enabled {
            return None;
        }
        if height.max(width) > self.max_dim {
            return None;
        }
        let key = PlanKey {
            dtype: PlanDType::F32,
            height,
            width,
            direction,
        };
        self.plans.get(&key).cloned()
    }
}

fn parse_small_sizes(max_dim: usize) -> Vec<usize> {
    if let Ok(spec) = env::var("RUSTFFT_SMALL_SIZES") {
        let mut sizes: Vec<usize> = spec
            .split(',')
            .filter_map(|token| token.trim().parse::<usize>().ok())
            .filter(|&size| size > 0 && size <= max_dim)
            .collect();
        sizes.sort_unstable();
        sizes.dedup();
        return sizes;
    }

    let mut size = 32usize;
    let mut sizes = Vec::new();
    while size <= max_dim {
        sizes.push(size);
        size *= 2;
    }
    sizes
}
