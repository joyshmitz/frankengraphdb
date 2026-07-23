//! Deterministic insertion-only Count-Min frequency sketches.
//!
//! The sketch is deliberately explicit about the operation it cannot support:
//! deletion returns [`CountMinError::RebuildRequired`] without changing state.
//! This prevents an advisory frequency estimate from silently drifting after a
//! property or edge is removed.

use core::fmt;
use core::hash::Hasher;
use fgdb_collections::hash_table::SeededHasher;
use std::collections::TryReserveError;

/// Conservative default ceiling for one sketch's counter matrix.
pub const DEFAULT_MAX_CELLS: usize = 16 * 1024 * 1024;

/// Complete profile governing shape, hashing, and resource bounds.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CountMinProfile {
    /// Counters in each row.
    pub width: usize,
    /// Independently seeded rows.
    pub depth: usize,
    /// Explicit deterministic root seed.
    pub seed: u64,
    /// Maximum accepted sum of observation weights.
    pub max_total_weight: u64,
    /// Maximum allocated counters.
    pub max_cells: usize,
}

impl CountMinProfile {
    /// Constructs a profile with the crate's conservative cell ceiling.
    #[must_use]
    pub const fn new(width: usize, depth: usize, seed: u64, max_total_weight: u64) -> Self {
        Self {
            width,
            depth,
            seed,
            max_total_weight,
            max_cells: DEFAULT_MAX_CELLS,
        }
    }
}

/// Typed failure from construction or a state transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CountMinError {
    /// Width and depth must both be nonzero.
    EmptyDimension {
        /// Rejected width.
        width: usize,
        /// Rejected depth.
        depth: usize,
    },
    /// Multiplying width by depth overflowed.
    CellCountOverflow,
    /// The configured matrix exceeds its explicit resource ceiling.
    CellLimitExceeded {
        /// Requested counters.
        requested: usize,
        /// Configured counter ceiling.
        limit: usize,
    },
    /// The allocator rejected the checked matrix reservation.
    AllocationFailed {
        /// Requested counters.
        requested: usize,
    },
    /// An update or merge would exceed a counter or total-weight bound.
    WeightOverflow,
    /// Merge operands use different complete profiles.
    ProfileMismatch,
    /// This profile cannot subtract observations exactly.
    RebuildRequired {
        /// Weight the caller wanted to remove.
        requested_weight: u64,
    },
}

impl fmt::Display for CountMinError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::EmptyDimension { width, depth } => {
                write!(
                    formatter,
                    "Count-Min dimensions must be nonzero, got {width}x{depth}"
                )
            }
            Self::CellCountOverflow => {
                formatter.write_str("Count-Min counter count overflows usize")
            }
            Self::CellLimitExceeded { requested, limit } => write!(
                formatter,
                "Count-Min requires {requested} counters, configured limit is {limit}"
            ),
            Self::AllocationFailed { requested } => {
                write!(
                    formatter,
                    "could not reserve {requested} Count-Min counters"
                )
            }
            Self::WeightOverflow => {
                formatter.write_str("Count-Min weight transition exceeds its exact bound")
            }
            Self::ProfileMismatch => {
                formatter.write_str("cannot merge Count-Min sketches with different profiles")
            }
            Self::RebuildRequired { requested_weight } => write!(
                formatter,
                "Count-Min cannot remove weight {requested_weight}; rebuild is required"
            ),
        }
    }
}

impl std::error::Error for CountMinError {}

/// Canonical logical state borrowed from a sketch.
///
/// Counters are row-major. Equal states have identical profiles, total weight,
/// and counter slices.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CountMinState<'sketch> {
    /// Complete behavior and hashing profile.
    pub profile: CountMinProfile,
    /// Sum of accepted observation weights.
    pub total_weight: u64,
    /// Row-major counter matrix.
    pub counters: &'sketch [u64],
}

/// Mergeable frequency upper-bound summary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CountMinSketch {
    profile: CountMinProfile,
    counters: Vec<u64>,
    total_weight: u64,
}

impl CountMinSketch {
    /// Allocates an all-zero matrix after validating every resource bound.
    pub fn try_new(profile: CountMinProfile) -> Result<Self, CountMinError> {
        if profile.width == 0 || profile.depth == 0 {
            return Err(CountMinError::EmptyDimension {
                width: profile.width,
                depth: profile.depth,
            });
        }
        let cell_count = profile
            .width
            .checked_mul(profile.depth)
            .ok_or(CountMinError::CellCountOverflow)?;
        if cell_count > profile.max_cells {
            return Err(CountMinError::CellLimitExceeded {
                requested: cell_count,
                limit: profile.max_cells,
            });
        }

        let mut counters = Vec::new();
        counters
            .try_reserve_exact(cell_count)
            .map_err(|_: TryReserveError| CountMinError::AllocationFailed {
                requested: cell_count,
            })?;
        counters.resize(cell_count, 0);
        Ok(Self {
            profile,
            counters,
            total_weight: 0,
        })
    }

    /// Returns the complete immutable profile.
    #[must_use]
    pub const fn profile(&self) -> CountMinProfile {
        self.profile
    }

    /// Returns the sum of accepted observation weights.
    #[must_use]
    pub const fn total_weight(&self) -> u64 {
        self.total_weight
    }

    /// Returns the canonical row-major state.
    #[must_use]
    pub fn canonical_state(&self) -> CountMinState<'_> {
        CountMinState {
            profile: self.profile,
            total_weight: self.total_weight,
            counters: &self.counters,
        }
    }

    /// Adds `weight` to a canonical byte key.
    ///
    /// The transition validates the total and every addressed counter before
    /// changing any state, so a typed failure leaves the sketch unchanged.
    pub fn try_observe(&mut self, key: &[u8], weight: u64) -> Result<(), CountMinError> {
        let next_total = self
            .total_weight
            .checked_add(weight)
            .filter(|total| *total <= self.profile.max_total_weight)
            .ok_or(CountMinError::WeightOverflow)?;

        for row in 0..self.profile.depth {
            let index = self.counter_index(row, key);
            self.counters[index]
                .checked_add(weight)
                .ok_or(CountMinError::WeightOverflow)?;
        }
        for row in 0..self.profile.depth {
            let index = self.counter_index(row, key);
            self.counters[index] += weight;
        }
        self.total_weight = next_total;
        Ok(())
    }

    /// Returns the Count-Min upper-bound estimate for `key`.
    #[must_use]
    pub fn estimate(&self, key: &[u8]) -> u64 {
        let mut estimate = u64::MAX;
        for row in 0..self.profile.depth {
            estimate = estimate.min(self.counters[self.counter_index(row, key)]);
        }
        estimate
    }

    /// Rejects deletion without mutating the insertion-only sketch.
    pub const fn try_remove(&mut self, _key: &[u8], weight: u64) -> Result<(), CountMinError> {
        Err(CountMinError::RebuildRequired {
            requested_weight: weight,
        })
    }

    /// Merges another sketch with the identical complete profile.
    ///
    /// Overflow checks cover the full matrix before either operand changes.
    pub fn try_merge(&mut self, other: &Self) -> Result<(), CountMinError> {
        if self.profile != other.profile {
            return Err(CountMinError::ProfileMismatch);
        }
        let next_total = self
            .total_weight
            .checked_add(other.total_weight)
            .filter(|total| *total <= self.profile.max_total_weight)
            .ok_or(CountMinError::WeightOverflow)?;
        for (&left, &right) in self.counters.iter().zip(&other.counters) {
            left.checked_add(right)
                .ok_or(CountMinError::WeightOverflow)?;
        }
        for (left, &right) in self.counters.iter_mut().zip(&other.counters) {
            *left += right;
        }
        self.total_weight = next_total;
        Ok(())
    }

    fn counter_index(&self, row: usize, key: &[u8]) -> usize {
        let mut hasher = SeededHasher::new(self.profile.seed);
        hasher.write_u64(row as u64);
        hasher.write(key);
        let width = self.profile.width as u64;
        row * self.profile.width + (hasher.finish() % width) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> CountMinProfile {
        CountMinProfile {
            width: 256,
            depth: 5,
            seed: 0x434d_534b_4554_4348,
            max_total_weight: 1_000_000,
            max_cells: 2_000,
        }
    }

    fn sketch() -> CountMinSketch {
        CountMinSketch::try_new(profile()).expect("bounded profile")
    }

    #[test]
    fn construction_enforces_shape_and_cell_ceiling() {
        assert_eq!(
            CountMinSketch::try_new(CountMinProfile::new(0, 3, 1, 10)),
            Err(CountMinError::EmptyDimension { width: 0, depth: 3 })
        );
        assert_eq!(
            CountMinSketch::try_new(CountMinProfile {
                width: usize::MAX,
                depth: 2,
                seed: 1,
                max_total_weight: 10,
                max_cells: usize::MAX,
            }),
            Err(CountMinError::CellCountOverflow)
        );
        assert_eq!(
            CountMinSketch::try_new(CountMinProfile {
                width: 8,
                depth: 3,
                seed: 1,
                max_total_weight: 10,
                max_cells: 23,
            }),
            Err(CountMinError::CellLimitExceeded {
                requested: 24,
                limit: 23,
            })
        );
    }

    #[test]
    fn estimates_never_understate_observed_weight() {
        let mut sketch = sketch();
        let observations = [
            (&b"person"[..], 11),
            (&b"knows"[..], 7),
            (&b"person"[..], 13),
            (&b"city"[..], 5),
        ];
        for (key, weight) in observations {
            sketch.try_observe(key, weight).expect("bounded update");
        }
        assert!(sketch.estimate(b"person") >= 24);
        assert!(sketch.estimate(b"knows") >= 7);
        assert!(sketch.estimate(b"city") >= 5);
        assert_eq!(sketch.total_weight(), 36);
    }

    #[test]
    fn observation_order_has_identical_canonical_state() {
        let observations = [
            (&b"alpha"[..], 3),
            (&b"beta"[..], 5),
            (&b"gamma"[..], 7),
            (&b"alpha"[..], 11),
        ];
        let mut forward = sketch();
        let mut reverse = sketch();
        for &(key, weight) in &observations {
            forward.try_observe(key, weight).expect("bounded update");
        }
        for &(key, weight) in observations.iter().rev() {
            reverse.try_observe(key, weight).expect("bounded update");
        }
        assert_eq!(forward.canonical_state(), reverse.canonical_state());
    }

    #[test]
    fn merge_is_commutative_and_associative_for_identical_profiles() {
        fn part(entries: &[(&[u8], u64)]) -> CountMinSketch {
            let mut value = sketch();
            for &(key, weight) in entries {
                value.try_observe(key, weight).expect("bounded update");
            }
            value
        }

        let a = part(&[(b"a", 2), (b"d", 3)]);
        let b = part(&[(b"b", 5), (b"a", 7)]);
        let c = part(&[(b"c", 11), (b"d", 13)]);

        let mut left = a.clone();
        left.try_merge(&b).expect("matching profile");
        let mut right = b.clone();
        right.try_merge(&a).expect("matching profile");
        assert_eq!(left, right);

        let mut ab_c = a.clone();
        ab_c.try_merge(&b).expect("matching profile");
        ab_c.try_merge(&c).expect("matching profile");
        let mut bc = b;
        bc.try_merge(&c).expect("matching profile");
        let mut a_bc = a;
        a_bc.try_merge(&bc).expect("matching profile");
        assert_eq!(ab_c, a_bc);
    }

    #[test]
    fn deletion_and_profile_mismatch_leave_state_unchanged() {
        let mut value = sketch();
        value.try_observe(b"edge", 9).expect("bounded update");
        let before = value.clone();
        assert_eq!(
            value.try_remove(b"edge", 4),
            Err(CountMinError::RebuildRequired {
                requested_weight: 4,
            })
        );
        assert_eq!(value, before);

        let mut other_profile = profile();
        other_profile.seed ^= 1;
        let other = CountMinSketch::try_new(other_profile).expect("bounded profile");
        assert_eq!(value.try_merge(&other), Err(CountMinError::ProfileMismatch));
        assert_eq!(value, before);
    }

    #[test]
    fn overflow_is_atomic() {
        let mut bounded = CountMinSketch::try_new(CountMinProfile {
            max_total_weight: 10,
            ..profile()
        })
        .expect("bounded profile");
        bounded.try_observe(b"x", 9).expect("within total bound");
        let before = bounded.clone();
        assert_eq!(
            bounded.try_observe(b"x", 2),
            Err(CountMinError::WeightOverflow)
        );
        assert_eq!(bounded, before);
    }
}
