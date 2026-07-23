//! Deterministic mergeable distinct-count sketches.
//!
//! The sketch is an insertion-only HyperLogLog-style register array.  Its
//! complete profile fixes precision, hashing seed, and the allocation ceiling;
//! register order is canonical; merge is lane-wise maximum; and deletion is a
//! typed rebuild request.  Estimation uses integer fixed-point arithmetic only,
//! so neither canonical state nor reported estimates depend on host floating
//! point behavior.

use core::fmt;
use core::hash::Hasher;
use fgdb_collections::hash_table::SeededHasher;
use std::collections::TryReserveError;

/// Smallest supported register-index precision.
pub const MIN_PRECISION: u8 = 4;

/// Largest supported register-index precision.
pub const MAX_PRECISION: u8 = 20;

/// Conservative default ceiling of one byte per register.
pub const DEFAULT_MAX_REGISTERS: usize = 1 << MAX_PRECISION;

/// Fraction bits carried by [`DistinctEstimate::scaled`].
pub const ESTIMATE_FRACTION_BITS: u32 = 32;

const DISTINCT_HASH_DOMAIN: u64 = 0x4647_4442_4449_5354;
const LN_2_Q64: u128 = 0xb172_17f7_d1cf_79ab;

/// Complete hashing, precision, and resource profile.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DistinctProfile {
    /// High hash bits used as the canonical register index.
    pub precision: u8,
    /// Explicit deterministic root seed.
    pub seed: u64,
    /// Maximum register bytes this profile may allocate.
    pub max_registers: usize,
}

impl DistinctProfile {
    /// Constructs a profile with the crate's conservative register ceiling.
    #[must_use]
    pub const fn new(precision: u8, seed: u64) -> Self {
        Self {
            precision,
            seed,
            max_registers: DEFAULT_MAX_REGISTERS,
        }
    }
}

/// Typed construction, merge, or deletion failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DistinctError {
    /// Precision lies outside the implemented deterministic profile family.
    PrecisionOutOfRange {
        /// Rejected precision.
        requested: u8,
        /// Inclusive minimum precision.
        minimum: u8,
        /// Inclusive maximum precision.
        maximum: u8,
    },
    /// Computing `2^precision` did not fit the target's address space.
    RegisterCountOverflow {
        /// Rejected precision.
        precision: u8,
    },
    /// The canonical register array exceeds the explicit resource ceiling.
    RegisterLimitExceeded {
        /// Required registers.
        requested: usize,
        /// Configured ceiling.
        limit: usize,
    },
    /// The allocator rejected the checked register reservation.
    AllocationFailed {
        /// Required registers.
        requested: usize,
    },
    /// Merge operands use different complete profiles.
    ProfileMismatch,
    /// A register array is not canonical for its declared precision.
    NonCanonicalRegisters {
        /// First invalid register index, or the first missing index on a shape
        /// mismatch.
        index: usize,
        /// Observed register value, with zero used for a missing lane.
        value: u8,
        /// Largest legal rank for the profile.
        maximum: u8,
    },
    /// Exact deletion is unavailable for max-register sketches.
    RebuildRequired,
}

impl fmt::Display for DistinctError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::PrecisionOutOfRange {
                requested,
                minimum,
                maximum,
            } => write!(
                formatter,
                "distinct-sketch precision {requested} is outside {minimum}..={maximum}"
            ),
            Self::RegisterCountOverflow { precision } => write!(
                formatter,
                "distinct-sketch register count for precision {precision} overflows usize"
            ),
            Self::RegisterLimitExceeded { requested, limit } => write!(
                formatter,
                "distinct sketch requires {requested} registers, configured limit is {limit}"
            ),
            Self::AllocationFailed { requested } => {
                write!(
                    formatter,
                    "could not reserve {requested} distinct-sketch registers"
                )
            }
            Self::ProfileMismatch => {
                formatter.write_str("cannot merge distinct sketches with different profiles")
            }
            Self::NonCanonicalRegisters {
                index,
                value,
                maximum,
            } => write!(
                formatter,
                "distinct-sketch register {index} has rank {value}, maximum is {maximum}"
            ),
            Self::RebuildRequired => {
                formatter.write_str("distinct sketch cannot delete exactly; rebuild is required")
            }
        }
    }
}

impl std::error::Error for DistinctError {}

/// Canonical logical state borrowed from a sketch.
///
/// Registers are ordered by the unsigned value of the high `precision` hash
/// bits.  Every register is in `0..=65-precision`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DistinctState<'sketch> {
    /// Complete behavior, hashing, and resource profile.
    pub profile: DistinctProfile,
    /// Canonically indexed register ranks.
    pub registers: &'sketch [u8],
}

/// Estimator branch selected from canonical register state.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DistinctEstimateMethod {
    /// Empty-register small-range correction, `m * ln(m / zeroes)`.
    LinearCounting,
    /// Bias-constant harmonic register estimator.
    RawHarmonic,
}

/// Deterministic Q32 cardinality estimate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DistinctEstimate {
    /// Estimate multiplied by `2^ESTIMATE_FRACTION_BITS`.
    pub scaled: u128,
    /// Estimator branch used for this register state.
    pub method: DistinctEstimateMethod,
    /// Number of zero registers observed by the estimator.
    pub zero_registers: usize,
}

impl DistinctEstimate {
    /// Integer estimate rounded to nearest, saturating at `u64::MAX`.
    #[must_use]
    pub fn rounded(self) -> u64 {
        let rounded = self
            .scaled
            .saturating_add(1_u128 << (ESTIMATE_FRACTION_BITS - 1))
            >> ESTIMATE_FRACTION_BITS;
        if rounded > u128::from(u64::MAX) {
            u64::MAX
        } else {
            rounded as u64
        }
    }

    /// Integer floor of the estimate, saturating at `u64::MAX`.
    #[must_use]
    pub fn floor(self) -> u64 {
        let floor = self.scaled >> ESTIMATE_FRACTION_BITS;
        if floor > u128::from(u64::MAX) {
            u64::MAX
        } else {
            floor as u64
        }
    }
}

/// Mergeable deterministic distinct-count summary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DistinctSketch {
    profile: DistinctProfile,
    registers: Vec<u8>,
}

impl DistinctSketch {
    /// Allocates a zeroed canonical register array after checking all bounds.
    pub fn try_new(profile: DistinctProfile) -> Result<Self, DistinctError> {
        if !(MIN_PRECISION..=MAX_PRECISION).contains(&profile.precision) {
            return Err(DistinctError::PrecisionOutOfRange {
                requested: profile.precision,
                minimum: MIN_PRECISION,
                maximum: MAX_PRECISION,
            });
        }
        let register_count = 1_usize.checked_shl(u32::from(profile.precision)).ok_or(
            DistinctError::RegisterCountOverflow {
                precision: profile.precision,
            },
        )?;
        if register_count > profile.max_registers {
            return Err(DistinctError::RegisterLimitExceeded {
                requested: register_count,
                limit: profile.max_registers,
            });
        }

        let mut registers = Vec::new();
        registers
            .try_reserve_exact(register_count)
            .map_err(|_: TryReserveError| DistinctError::AllocationFailed {
                requested: register_count,
            })?;
        registers.resize(register_count, 0);
        Ok(Self { profile, registers })
    }

    /// Complete immutable profile.
    #[must_use]
    pub const fn profile(&self) -> DistinctProfile {
        self.profile
    }

    /// Number of canonical registers.
    #[must_use]
    pub fn register_count(&self) -> usize {
        self.registers.len()
    }

    /// Largest legal rank under this profile.
    #[must_use]
    pub const fn maximum_rank(&self) -> u8 {
        maximum_rank(self.profile.precision)
    }

    /// Canonical logical state.
    #[must_use]
    pub fn canonical_state(&self) -> DistinctState<'_> {
        DistinctState {
            profile: self.profile,
            registers: &self.registers,
        }
    }

    /// Observes a canonical byte key.
    ///
    /// Returns `true` exactly when one canonical register increased.
    pub fn observe(&mut self, key: &[u8]) -> bool {
        self.observe_hash(hash_key(self.profile.seed, key))
    }

    /// Rejects deletion without changing the insertion-only sketch.
    pub const fn try_remove(&mut self, _key: &[u8]) -> Result<(), DistinctError> {
        Err(DistinctError::RebuildRequired)
    }

    /// Merges another profile-identical sketch with lane-wise maximum.
    ///
    /// Both arrays are validated before mutation, so every failure leaves the
    /// receiver byte-for-byte unchanged.
    pub fn try_merge(&mut self, other: &Self) -> Result<(), DistinctError> {
        if self.profile != other.profile {
            return Err(DistinctError::ProfileMismatch);
        }
        self.validate_canonical_shape()?;
        other.validate_canonical_shape()?;
        for (left, &right) in self.registers.iter_mut().zip(&other.registers) {
            *left = (*left).max(right);
        }
        Ok(())
    }

    /// Returns a deterministic Q32 estimate and its selected estimator branch.
    #[must_use]
    pub fn estimate_fixed(&self) -> DistinctEstimate {
        let register_count = self.registers.len() as u64;
        let zero_registers = self
            .registers
            .iter()
            .filter(|&&register| register == 0)
            .count();
        let raw = self.raw_estimate_q32(register_count);
        let small_range_threshold = u128::from(register_count).saturating_mul(5_u128 << 31);
        if zero_registers != 0 && raw <= small_range_threshold {
            let log_q32 = ln_ratio_q32(register_count, zero_registers as u64);
            DistinctEstimate {
                scaled: u128::from(register_count).saturating_mul(log_q32),
                method: DistinctEstimateMethod::LinearCounting,
                zero_registers,
            }
        } else {
            DistinctEstimate {
                scaled: raw,
                method: DistinctEstimateMethod::RawHarmonic,
                zero_registers,
            }
        }
    }

    /// Returns the fixed-point estimate rounded to the nearest integer.
    #[must_use]
    pub fn estimate(&self) -> u64 {
        self.estimate_fixed().rounded()
    }

    fn observe_hash(&mut self, hash: u64) -> bool {
        let (index, rank) = index_and_rank(hash, self.profile.precision);
        let Some(register) = self.registers.get_mut(index) else {
            return false;
        };
        if rank > *register {
            *register = rank;
            true
        } else {
            false
        }
    }

    fn validate_canonical_shape(&self) -> Result<(), DistinctError> {
        let expected = 1_usize
            .checked_shl(u32::from(self.profile.precision))
            .ok_or(DistinctError::RegisterCountOverflow {
                precision: self.profile.precision,
            })?;
        let maximum = self.maximum_rank();
        if self.registers.len() != expected {
            return Err(DistinctError::NonCanonicalRegisters {
                index: self.registers.len().min(expected),
                value: 0,
                maximum,
            });
        }
        for (index, &value) in self.registers.iter().enumerate() {
            if value > maximum {
                return Err(DistinctError::NonCanonicalRegisters {
                    index,
                    value,
                    maximum,
                });
            }
        }
        Ok(())
    }

    fn raw_estimate_q32(&self, register_count: u64) -> u128 {
        let maximum = self.maximum_rank();
        let mut harmonic_scaled = 0_u128;
        for &register in &self.registers {
            let shift = u32::from(maximum.saturating_sub(register));
            harmonic_scaled += 1_u128 << shift;
        }
        let numerator = alpha_q32(register_count)
            .saturating_mul(u128::from(register_count))
            .saturating_mul(u128::from(register_count))
            .saturating_mul(1_u128 << u32::from(maximum));
        numerator / harmonic_scaled.max(1)
    }
}

const fn maximum_rank(precision: u8) -> u8 {
    65 - precision
}

fn hash_key(seed: u64, key: &[u8]) -> u64 {
    let mut hasher = SeededHasher::new(seed);
    hasher.write_u64(DISTINCT_HASH_DOMAIN);
    hasher.write_u64(key.len() as u64);
    hasher.write(key);
    hasher.finish()
}

fn index_and_rank(hash: u64, precision: u8) -> (usize, u8) {
    let shift = u32::from(64 - precision);
    let index = (hash >> shift) as usize;
    let suffix = hash << u32::from(precision);
    let rank = suffix
        .leading_zeros()
        .saturating_add(1)
        .min(u32::from(maximum_rank(precision)));
    (index, rank as u8)
}

fn alpha_q32(register_count: u64) -> u128 {
    match register_count {
        16 => rounded_ratio_q32(673, 1_000),
        32 => rounded_ratio_q32(697, 1_000),
        64 => rounded_ratio_q32(709, 1_000),
        _ => rounded_ratio_q32(
            7_213_u128.saturating_mul(u128::from(register_count)),
            10_u128.saturating_mul(
                1_000_u128
                    .saturating_mul(u128::from(register_count))
                    .saturating_add(1_079),
            ),
        ),
    }
}

fn rounded_ratio_q32(numerator: u128, denominator: u128) -> u128 {
    numerator
        .saturating_mul(1_u128 << ESTIMATE_FRACTION_BITS)
        .saturating_add(denominator / 2)
        / denominator
}

fn ln_ratio_q32(numerator: u64, denominator: u64) -> u128 {
    if numerator <= denominator || denominator == 0 {
        return 0;
    }

    let mut reduced_denominator = denominator;
    let mut exponent = 0_u32;
    while reduced_denominator <= numerator / 2 {
        reduced_denominator *= 2;
        exponent += 1;
    }

    let numerator = u128::from(numerator);
    let reduced_denominator = u128::from(reduced_denominator);
    let z_q64 = ((numerator - reduced_denominator) << 64) / (numerator + reduced_denominator);
    let z_squared_q64 = multiply_q64(z_q64, z_q64);
    let mut term_q64 = z_q64;
    let mut series_q64 = term_q64;
    for iteration in 1_u128..=32 {
        term_q64 = multiply_q64(term_q64, z_squared_q64);
        if term_q64 == 0 {
            break;
        }
        series_q64 += term_q64 / (2 * iteration + 1);
    }
    let log_q64 = u128::from(exponent)
        .saturating_mul(LN_2_Q64)
        .saturating_add(series_q64.saturating_mul(2));
    log_q64.saturating_add(1_u128 << 31) >> 32
}

fn multiply_q64(left: u128, right: u128) -> u128 {
    left.saturating_mul(right) >> 64
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_MAX_REGISTERS, DistinctError, DistinctEstimateMethod, DistinctProfile,
        DistinctSketch, MAX_PRECISION, MIN_PRECISION, index_and_rank, maximum_rank,
    };

    fn profile() -> DistinctProfile {
        DistinctProfile {
            precision: 12,
            seed: 0x484c_4c44_4953_5431,
            max_registers: 1 << 12,
        }
    }

    fn sketch() -> Result<DistinctSketch, DistinctError> {
        DistinctSketch::try_new(profile())
    }

    fn populated(keys: &[u64]) -> Result<DistinctSketch, DistinctError> {
        let mut value = sketch()?;
        for key in keys {
            value.observe(&key.to_le_bytes());
        }
        Ok(value)
    }

    #[test]
    fn construction_enforces_precision_and_resource_ceiling() {
        assert_eq!(
            DistinctSketch::try_new(DistinctProfile::new(MIN_PRECISION - 1, 1)),
            Err(DistinctError::PrecisionOutOfRange {
                requested: MIN_PRECISION - 1,
                minimum: MIN_PRECISION,
                maximum: MAX_PRECISION,
            })
        );
        assert_eq!(
            DistinctSketch::try_new(DistinctProfile::new(MAX_PRECISION + 1, 1)),
            Err(DistinctError::PrecisionOutOfRange {
                requested: MAX_PRECISION + 1,
                minimum: MIN_PRECISION,
                maximum: MAX_PRECISION,
            })
        );
        assert_eq!(
            DistinctSketch::try_new(DistinctProfile {
                precision: 10,
                seed: 1,
                max_registers: 1_023,
            }),
            Err(DistinctError::RegisterLimitExceeded {
                requested: 1_024,
                limit: 1_023,
            })
        );
        assert_eq!(
            DistinctProfile::new(MAX_PRECISION, 7).max_registers,
            DEFAULT_MAX_REGISTERS
        );
    }

    #[test]
    fn index_and_rank_cover_first_last_and_zero_suffix_boundaries() {
        assert_eq!(index_and_rank(0, 4), (0, maximum_rank(4)));
        assert_eq!(
            index_and_rank(0xf000_0000_0000_0000, 4),
            (15, maximum_rank(4))
        );
        assert_eq!(index_and_rank(0x0800_0000_0000_0000, 4), (0, 1));
        assert_eq!(index_and_rank(0x0400_0000_0000_0000, 4), (0, 2));
        assert_eq!(
            index_and_rank(0xffff_f800_0000_0000, 20),
            ((1 << 20) - 1, 1)
        );
        assert_eq!(maximum_rank(MIN_PRECISION), 61);
        assert_eq!(maximum_rank(MAX_PRECISION), 45);
    }

    #[test]
    fn observation_is_canonical_and_order_duplicate_independent() -> Result<(), DistinctError> {
        let keys: Vec<_> = (0_u64..2_000).collect();
        let mut forward = populated(&keys)?;
        let reverse_keys: Vec<_> = keys.iter().rev().copied().collect();
        let mut reverse = populated(&reverse_keys)?;
        for key in &keys {
            forward.observe(&key.to_le_bytes());
            reverse.observe(&key.to_le_bytes());
        }
        assert_eq!(forward.canonical_state(), reverse.canonical_state());

        let same_seed = populated(&keys)?;
        assert_eq!(forward.canonical_state(), same_seed.canonical_state());
        let mut other_profile = profile();
        other_profile.seed ^= 1;
        let mut other = DistinctSketch::try_new(other_profile)?;
        for key in &keys {
            other.observe(&key.to_le_bytes());
        }
        assert_ne!(
            forward.canonical_state().registers,
            other.canonical_state().registers
        );
        Ok(())
    }

    #[test]
    fn merge_is_commutative_associative_and_idempotent() -> Result<(), DistinctError> {
        let a = populated(&(0_u64..2_000).collect::<Vec<_>>())?;
        let b = populated(&(1_000_u64..3_000).collect::<Vec<_>>())?;
        let c = populated(&(2_500_u64..4_000).collect::<Vec<_>>())?;

        let mut ab = a.clone();
        ab.try_merge(&b)?;
        let mut ba = b.clone();
        ba.try_merge(&a)?;
        assert_eq!(ab, ba);

        let mut ab_c = a.clone();
        ab_c.try_merge(&b)?;
        ab_c.try_merge(&c)?;
        let mut bc = b;
        bc.try_merge(&c)?;
        let mut a_bc = a.clone();
        a_bc.try_merge(&bc)?;
        assert_eq!(ab_c, a_bc);

        let before = a.clone();
        let mut idempotent = a.clone();
        idempotent.try_merge(&a)?;
        assert_eq!(idempotent, before);
        Ok(())
    }

    #[test]
    fn deletion_and_profile_mismatch_are_typed_and_atomic() -> Result<(), DistinctError> {
        let mut value = populated(&(0_u64..100).collect::<Vec<_>>())?;
        let before = value.clone();
        assert_eq!(
            value.try_remove(&42_u64.to_le_bytes()),
            Err(DistinctError::RebuildRequired)
        );
        assert_eq!(value, before);

        for mutate in [
            DistinctProfile {
                seed: profile().seed ^ 1,
                ..profile()
            },
            DistinctProfile {
                max_registers: profile().max_registers + 1,
                ..profile()
            },
            DistinctProfile {
                precision: profile().precision + 1,
                max_registers: 1 << (profile().precision + 1),
                ..profile()
            },
        ] {
            let other = DistinctSketch::try_new(mutate)?;
            assert_eq!(value.try_merge(&other), Err(DistinctError::ProfileMismatch));
            assert_eq!(value, before);
        }
        Ok(())
    }

    #[test]
    fn integer_estimates_track_named_synthetic_cardinalities() -> Result<(), DistinctError> {
        struct Case {
            name: &'static str,
            cardinality: u64,
            tolerance_parts_per_million: u64,
        }
        let cases = [
            Case {
                name: "empty",
                cardinality: 0,
                tolerance_parts_per_million: 0,
            },
            Case {
                name: "singleton",
                cardinality: 1,
                tolerance_parts_per_million: 0,
            },
            Case {
                name: "tiny-10",
                cardinality: 10,
                tolerance_parts_per_million: 100_000,
            },
            Case {
                name: "small-1k",
                cardinality: 1_000,
                tolerance_parts_per_million: 60_000,
            },
            Case {
                name: "medium-10k",
                cardinality: 10_000,
                tolerance_parts_per_million: 60_000,
            },
            Case {
                name: "large-100k",
                cardinality: 100_000,
                tolerance_parts_per_million: 60_000,
            },
        ];

        for case in cases {
            let mut value = DistinctSketch::try_new(DistinctProfile {
                precision: 14,
                seed: 0x5359_4e54_4845_5449,
                max_registers: 1 << 14,
            })?;
            for key in 0..case.cardinality {
                value.observe(&key.to_le_bytes());
            }
            let estimate = value.estimate();
            let error = estimate.abs_diff(case.cardinality);
            let permitted = case
                .cardinality
                .saturating_mul(case.tolerance_parts_per_million)
                / 1_000_000;
            assert!(
                error <= permitted,
                "{}: estimate {estimate}, truth {}, error {error}, permitted {permitted}",
                case.name,
                case.cardinality
            );
            let fixed = value.estimate_fixed();
            assert_eq!(fixed.rounded(), estimate);
            if case.cardinality == 0 {
                assert_eq!(fixed.method, DistinctEstimateMethod::LinearCounting);
                assert_eq!(fixed.scaled, 0);
            }
        }
        Ok(())
    }
}
