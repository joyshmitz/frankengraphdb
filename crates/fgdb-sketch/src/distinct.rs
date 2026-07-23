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
const CANONICAL_MAGIC: [u8; 8] = *b"FGDBDST1";
const CANONICAL_VERSION: u16 = 1;
const CANONICAL_HEADER_BYTES: usize = 8 + 2 + 1 + 1 + 8 + 8 + 8;

/// Stable hash algorithm named by a distinct-sketch profile.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum DistinctHashAlgorithm {
    /// Domain-separated [`SeededHasher`] stream frozen by canonical vectors.
    SeededHasherV1 = 1,
}

impl DistinctHashAlgorithm {
    const fn canonical_tag(self) -> u8 {
        self as u8
    }
}

/// Complete hashing, precision, and resource profile.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DistinctProfile {
    /// High hash bits used as the canonical register index.
    pub precision: u8,
    /// Stable hash algorithm included in profile identity.
    pub hash_algorithm: DistinctHashAlgorithm,
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
            hash_algorithm: DistinctHashAlgorithm::SeededHasherV1,
            seed,
            max_registers: DEFAULT_MAX_REGISTERS,
        }
    }
}

/// Caller-owned admission bounds for decoding one distinct sketch.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DistinctDecodeLimits {
    /// Maximum accepted canonical value length.
    pub max_encoded_bytes: usize,
    /// Maximum accepted encoded profile ceiling and register allocation.
    pub max_registers: usize,
}

impl DistinctDecodeLimits {
    /// Creates explicit caller-owned decode admission bounds.
    #[must_use]
    pub const fn new(max_encoded_bytes: usize, max_registers: usize) -> Self {
        Self {
            max_encoded_bytes,
            max_registers,
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

/// Caller-owned resource checked while admitting canonical distinct bytes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DistinctDecodeResource {
    /// Total canonical input bytes.
    EncodedBytes,
    /// Profile ceiling or materialized register count.
    Registers,
}

/// Strict canonical-codec failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DistinctCodecError {
    /// The in-memory or decoded sketch violates a construction invariant.
    State(DistinctError),
    /// A platform-sized profile field cannot be represented canonically.
    IntegerUnrepresentable,
    /// Computing the exact canonical value length overflowed.
    LengthOverflow,
    /// The allocator rejected the exact canonical byte reservation.
    AllocationFailed {
        /// Requested byte count.
        requested: usize,
    },
    /// The eight-byte format discriminator did not match.
    MagicMismatch {
        /// Bytes found at the canonical magic position.
        actual: [u8; 8],
    },
    /// The format version is not implemented.
    UnsupportedVersion {
        /// Version found in the input.
        actual: u16,
    },
    /// The encoded hash-algorithm discriminator is not implemented.
    UnsupportedHashAlgorithm {
        /// Discriminator found in the canonical profile.
        actual: u8,
    },
    /// The encoded complete profile does not equal the trusted expected profile.
    ProfileMismatch {
        /// Trusted profile supplied by the caller.
        expected: DistinctProfile,
        /// Profile decoded from the canonical header.
        actual: DistinctProfile,
    },
    /// A caller-owned decode admission bound was exceeded.
    DecodeLimitExceeded {
        /// Resource whose admission bound was exceeded.
        resource: DistinctDecodeResource,
        /// Encoded profile, state, or input value.
        actual: usize,
        /// Caller-owned maximum.
        maximum: usize,
    },
    /// Input ended before a complete field could be read.
    Truncated {
        /// Byte offset of the field or payload.
        offset: usize,
        /// Bytes needed at that offset.
        needed: usize,
        /// Bytes remaining at the offset.
        remaining: usize,
    },
    /// The encoded register count disagrees with the precision.
    RegisterCountMismatch {
        /// Register count implied by the encoded precision.
        expected: usize,
        /// Register count declared by the input.
        actual: usize,
    },
    /// Input contains bytes after the one canonical value.
    TrailingBytes {
        /// First trailing byte.
        offset: usize,
        /// Number of trailing bytes.
        remaining: usize,
    },
}

impl fmt::Display for DistinctCodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for DistinctCodecError {}

impl From<DistinctError> for DistinctCodecError {
    fn from(error: DistinctError) -> Self {
        Self::State(error)
    }
}

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
        let register_count = checked_register_count(profile)?;

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

    /// Encodes the complete profile and logical state into one canonical value.
    ///
    /// The representation uses fixed-width big-endian fields followed by the
    /// canonically indexed one-byte register array. Equal logical states
    /// therefore produce byte-identical values without relying on host word
    /// size.
    pub fn try_to_canonical_bytes(&self) -> Result<Vec<u8>, DistinctCodecError> {
        self.validate_canonical_shape()?;
        let canonical_max_registers = canonical_usize(self.profile.max_registers)?;
        let canonical_register_count = canonical_usize(self.registers.len())?;
        let encoded_len = CANONICAL_HEADER_BYTES
            .checked_add(self.registers.len())
            .ok_or(DistinctCodecError::LengthOverflow)?;

        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(encoded_len)
            .map_err(|_: TryReserveError| DistinctCodecError::AllocationFailed {
                requested: encoded_len,
            })?;
        bytes.extend_from_slice(&CANONICAL_MAGIC);
        push_u16(&mut bytes, CANONICAL_VERSION);
        bytes.push(self.profile.hash_algorithm.canonical_tag());
        bytes.push(self.profile.precision);
        push_u64(&mut bytes, self.profile.seed);
        push_u64(&mut bytes, canonical_max_registers);
        push_u64(&mut bytes, canonical_register_count);
        bytes.extend_from_slice(&self.registers);
        debug_assert_eq!(bytes.len(), encoded_len);
        Ok(bytes)
    }

    /// Decodes exactly one strict canonical value and revalidates every law.
    ///
    /// The complete header, declared shape, exact input length, and every
    /// register rank are validated before allocating the register array.
    pub fn try_from_canonical_bytes(
        bytes: &[u8],
        expected_profile: DistinctProfile,
        limits: DistinctDecodeLimits,
    ) -> Result<Self, DistinctCodecError> {
        enforce_decode_limit(
            DistinctDecodeResource::EncodedBytes,
            bytes.len(),
            limits.max_encoded_bytes,
        )?;
        let mut decoder = DistinctDecoder::new(bytes);
        let magic = decoder.read_array::<8>()?;
        if magic != CANONICAL_MAGIC {
            return Err(DistinctCodecError::MagicMismatch { actual: magic });
        }
        let version = decoder.read_u16()?;
        if version != CANONICAL_VERSION {
            return Err(DistinctCodecError::UnsupportedVersion { actual: version });
        }
        let hash_algorithm = decode_hash_algorithm(decoder.read_u8()?)?;
        let precision = decoder.read_u8()?;
        let seed = decoder.read_u64()?;
        let max_registers = decoded_usize(decoder.read_u64()?)?;
        let encoded_register_count = decoded_usize(decoder.read_u64()?)?;
        let profile = DistinctProfile {
            precision,
            hash_algorithm,
            seed,
            max_registers,
        };
        if profile != expected_profile {
            return Err(DistinctCodecError::ProfileMismatch {
                expected: expected_profile,
                actual: profile,
            });
        }
        enforce_decode_limit(
            DistinctDecodeResource::Registers,
            profile.max_registers,
            limits.max_registers,
        )?;
        let expected_register_count = checked_register_count(profile)?;
        if encoded_register_count != expected_register_count {
            return Err(DistinctCodecError::RegisterCountMismatch {
                expected: expected_register_count,
                actual: encoded_register_count,
            });
        }
        enforce_decode_limit(
            DistinctDecodeResource::Registers,
            encoded_register_count,
            limits.max_registers,
        )?;
        let expected_len = CANONICAL_HEADER_BYTES
            .checked_add(encoded_register_count)
            .ok_or(DistinctCodecError::LengthOverflow)?;
        if bytes.len() < expected_len {
            return Err(DistinctCodecError::Truncated {
                offset: decoder.offset,
                needed: encoded_register_count,
                remaining: bytes.len().saturating_sub(decoder.offset),
            });
        }
        if bytes.len() > expected_len {
            return Err(DistinctCodecError::TrailingBytes {
                offset: expected_len,
                remaining: bytes.len() - expected_len,
            });
        }

        let encoded_registers = decoder.take(encoded_register_count)?;
        decoder.finish()?;
        validate_registers(profile, encoded_registers)?;

        let mut sketch = Self::try_new(profile)?;
        sketch.registers.copy_from_slice(encoded_registers);
        Ok(sketch)
    }

    /// Observes a canonical byte key.
    ///
    /// Returns `true` exactly when one canonical register increased.
    pub fn observe(&mut self, key: &[u8]) -> bool {
        self.observe_hash(hash_key(
            self.profile.hash_algorithm,
            self.profile.seed,
            key,
        ))
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
        validate_registers(self.profile, &self.registers)
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

fn checked_register_count(profile: DistinctProfile) -> Result<usize, DistinctError> {
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
    Ok(register_count)
}

fn validate_registers(profile: DistinctProfile, registers: &[u8]) -> Result<(), DistinctError> {
    let expected = checked_register_count(profile)?;
    let maximum = maximum_rank(profile.precision);
    if registers.len() != expected {
        return Err(DistinctError::NonCanonicalRegisters {
            index: registers.len().min(expected),
            value: 0,
            maximum,
        });
    }
    for (index, &value) in registers.iter().enumerate() {
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

const fn maximum_rank(precision: u8) -> u8 {
    65 - precision
}

fn hash_key(algorithm: DistinctHashAlgorithm, seed: u64, key: &[u8]) -> u64 {
    match algorithm {
        DistinctHashAlgorithm::SeededHasherV1 => {
            let mut hasher = SeededHasher::new(seed);
            hasher.write_u64(DISTINCT_HASH_DOMAIN);
            hasher.write_u64(key.len() as u64);
            hasher.write(key);
            hasher.finish()
        }
    }
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

fn enforce_decode_limit(
    resource: DistinctDecodeResource,
    actual: usize,
    maximum: usize,
) -> Result<(), DistinctCodecError> {
    if actual > maximum {
        Err(DistinctCodecError::DecodeLimitExceeded {
            resource,
            actual,
            maximum,
        })
    } else {
        Ok(())
    }
}

fn canonical_usize(value: usize) -> Result<u64, DistinctCodecError> {
    u64::try_from(value).map_err(|_| DistinctCodecError::IntegerUnrepresentable)
}

fn decoded_usize(value: u64) -> Result<usize, DistinctCodecError> {
    usize::try_from(value).map_err(|_| DistinctCodecError::IntegerUnrepresentable)
}

fn decode_hash_algorithm(actual: u8) -> Result<DistinctHashAlgorithm, DistinctCodecError> {
    match actual {
        value if value == DistinctHashAlgorithm::SeededHasherV1.canonical_tag() => {
            Ok(DistinctHashAlgorithm::SeededHasherV1)
        }
        actual => Err(DistinctCodecError::UnsupportedHashAlgorithm { actual }),
    }
}

fn push_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_be_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_be_bytes());
}

struct DistinctDecoder<'bytes> {
    bytes: &'bytes [u8],
    offset: usize,
}

impl<'bytes> DistinctDecoder<'bytes> {
    const fn new(bytes: &'bytes [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, needed: usize) -> Result<&'bytes [u8], DistinctCodecError> {
        let end = self
            .offset
            .checked_add(needed)
            .ok_or(DistinctCodecError::LengthOverflow)?;
        let Some(value) = self.bytes.get(self.offset..end) else {
            return Err(DistinctCodecError::Truncated {
                offset: self.offset,
                needed,
                remaining: self.bytes.len().saturating_sub(self.offset),
            });
        };
        self.offset = end;
        Ok(value)
    }

    fn read_array<const LENGTH: usize>(&mut self) -> Result<[u8; LENGTH], DistinctCodecError> {
        let source = self.take(LENGTH)?;
        let mut value = [0_u8; LENGTH];
        value.copy_from_slice(source);
        Ok(value)
    }

    fn read_u8(&mut self) -> Result<u8, DistinctCodecError> {
        Ok(self.read_array::<1>()?[0])
    }

    fn read_u16(&mut self) -> Result<u16, DistinctCodecError> {
        Ok(u16::from_be_bytes(self.read_array::<2>()?))
    }

    fn read_u64(&mut self) -> Result<u64, DistinctCodecError> {
        Ok(u64::from_be_bytes(self.read_array::<8>()?))
    }

    fn finish(self) -> Result<(), DistinctCodecError> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(DistinctCodecError::TrailingBytes {
                offset: self.offset,
                remaining: self.bytes.len() - self.offset,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CANONICAL_HEADER_BYTES, CANONICAL_MAGIC, DEFAULT_MAX_REGISTERS, DistinctCodecError,
        DistinctDecodeLimits, DistinctDecodeResource, DistinctError, DistinctEstimateMethod,
        DistinctHashAlgorithm, DistinctProfile, DistinctSketch, MAX_PRECISION, MIN_PRECISION,
        hash_key, index_and_rank, maximum_rank,
    };

    const VERSION_OFFSET: usize = CANONICAL_MAGIC.len();
    const HASH_ALGORITHM_OFFSET: usize = VERSION_OFFSET + 2;
    const PRECISION_OFFSET: usize = HASH_ALGORITHM_OFFSET + 1;
    const SEED_OFFSET: usize = PRECISION_OFFSET + 1;
    const MAX_REGISTERS_OFFSET: usize = SEED_OFFSET + 8;
    const REGISTER_COUNT_OFFSET: usize = MAX_REGISTERS_OFFSET + 8;

    fn profile() -> DistinctProfile {
        DistinctProfile {
            precision: 12,
            hash_algorithm: DistinctHashAlgorithm::SeededHasherV1,
            seed: 0x484c_4c44_4953_5431,
            max_registers: 1 << 12,
        }
    }

    fn sketch() -> Result<DistinctSketch, DistinctError> {
        DistinctSketch::try_new(profile())
    }

    fn decode_limits(profile: DistinctProfile) -> DistinctDecodeLimits {
        DistinctDecodeLimits::new(usize::MAX, profile.max_registers)
    }

    fn read_fixture(
        bytes: &[u8],
        expected_profile: DistinctProfile,
    ) -> Result<DistinctSketch, DistinctCodecError> {
        DistinctSketch::try_from_canonical_bytes(
            bytes,
            expected_profile,
            decode_limits(expected_profile),
        )
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
                hash_algorithm: DistinctHashAlgorithm::SeededHasherV1,
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
        let frozen_hash = hash_key(
            DistinctHashAlgorithm::SeededHasherV1,
            profile().seed,
            b"alpha",
        );
        assert_eq!(frozen_hash, 0xa32f_ef99_88df_3345);
        assert_eq!(index_and_rank(frozen_hash, profile().precision), (2_610, 1));
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
    fn canonical_codec_round_trips_full_profile_and_collapses_order() -> Result<(), DistinctError> {
        let keys: Vec<_> = (0_u64..2_000).collect();
        let forward = populated(&keys)?;
        let reverse_keys: Vec<_> = keys.iter().rev().copied().collect();
        let reverse = populated(&reverse_keys)?;

        let forward_bytes = forward
            .try_to_canonical_bytes()
            .expect("valid sketch must encode");
        let reverse_bytes = reverse
            .try_to_canonical_bytes()
            .expect("valid sketch must encode");
        assert_eq!(forward_bytes, reverse_bytes);
        assert_eq!(&forward_bytes[..VERSION_OFFSET], b"FGDBDST1");
        assert_eq!(
            &forward_bytes[VERSION_OFFSET..HASH_ALGORITHM_OFFSET],
            &1_u16.to_be_bytes()
        );
        assert_eq!(
            forward_bytes[HASH_ALGORITHM_OFFSET],
            DistinctHashAlgorithm::SeededHasherV1.canonical_tag()
        );
        assert_eq!(forward_bytes[PRECISION_OFFSET], profile().precision);
        assert_eq!(
            &forward_bytes[SEED_OFFSET..MAX_REGISTERS_OFFSET],
            &profile().seed.to_be_bytes()
        );
        assert_eq!(
            &forward_bytes[MAX_REGISTERS_OFFSET..REGISTER_COUNT_OFFSET],
            &(profile().max_registers as u64).to_be_bytes()
        );
        assert_eq!(
            &forward_bytes[REGISTER_COUNT_OFFSET..CANONICAL_HEADER_BYTES],
            &(forward.register_count() as u64).to_be_bytes()
        );

        let decoded =
            read_fixture(&forward_bytes, forward.profile()).expect("canonical sketch must decode");
        assert_eq!(decoded, forward);
        assert_eq!(decoded.profile(), profile());
        assert_eq!(decoded.canonical_state(), forward.canonical_state());
        assert_eq!(
            decoded
                .try_to_canonical_bytes()
                .expect("decoded sketch must re-encode"),
            forward_bytes
        );

        let mut looser_profile = profile();
        looser_profile.max_registers += 1;
        let mut looser = DistinctSketch::try_new(looser_profile)?;
        for key in &keys {
            looser.observe(&key.to_le_bytes());
        }
        assert_eq!(
            looser.canonical_state().registers,
            forward.canonical_state().registers
        );
        let looser_bytes = looser
            .try_to_canonical_bytes()
            .expect("valid alternate profile must encode");
        assert_ne!(looser_bytes, forward_bytes);
        assert_eq!(
            read_fixture(&looser_bytes, looser_profile)
                .expect("complete alternate profile must decode")
                .profile(),
            looser_profile
        );
        Ok(())
    }

    #[test]
    fn canonical_decoder_requires_exact_profile_and_caller_owned_limits() {
        let value = sketch().expect("valid profile");
        let encoded = value.try_to_canonical_bytes().expect("valid sketch");
        let expected_profile = value.profile();
        let exact_limits = DistinctDecodeLimits::new(encoded.len(), expected_profile.max_registers);
        assert_eq!(
            DistinctSketch::try_from_canonical_bytes(&encoded, expected_profile, exact_limits)
                .expect("exact admission bounds accept the value"),
            value
        );

        let wrong_profile = DistinctProfile {
            seed: expected_profile.seed ^ 1,
            ..expected_profile
        };
        assert_eq!(
            DistinctSketch::try_from_canonical_bytes(&encoded, wrong_profile, exact_limits),
            Err(DistinctCodecError::ProfileMismatch {
                expected: wrong_profile,
                actual: expected_profile,
            })
        );
        assert_eq!(
            DistinctSketch::try_from_canonical_bytes(
                &encoded,
                expected_profile,
                DistinctDecodeLimits {
                    max_encoded_bytes: encoded.len() - 1,
                    ..exact_limits
                },
            ),
            Err(DistinctCodecError::DecodeLimitExceeded {
                resource: DistinctDecodeResource::EncodedBytes,
                actual: encoded.len(),
                maximum: encoded.len() - 1,
            })
        );
        assert_eq!(
            DistinctSketch::try_from_canonical_bytes(
                &encoded,
                expected_profile,
                DistinctDecodeLimits {
                    max_registers: expected_profile.max_registers - 1,
                    ..exact_limits
                },
            ),
            Err(DistinctCodecError::DecodeLimitExceeded {
                resource: DistinctDecodeResource::Registers,
                actual: expected_profile.max_registers,
                maximum: expected_profile.max_registers - 1,
            })
        );
    }

    #[test]
    fn canonical_codec_round_trips_precision_boundaries() {
        for precision in [MIN_PRECISION, 8, 12, MAX_PRECISION] {
            let profile =
                DistinctProfile::new(precision, 0x434f_4445_4350_0000 | u64::from(precision));
            let mut value = DistinctSketch::try_new(profile).expect("supported precision");
            value.observe(b"alpha");
            value.observe(b"omega");
            let encoded = value
                .try_to_canonical_bytes()
                .expect("bounded sketch must encode");
            assert_eq!(encoded.len(), CANONICAL_HEADER_BYTES + (1 << precision));
            let decoded = read_fixture(&encoded, profile).expect("canonical boundary sketch");
            assert_eq!(decoded, value);
        }
    }

    #[test]
    fn canonical_codec_preserves_merge_and_deletion_laws() -> Result<(), DistinctError> {
        let a = populated(&(0_u64..1_000).collect::<Vec<_>>())?;
        let b = populated(&(500_u64..1_500).collect::<Vec<_>>())?;
        let c = populated(&(1_200_u64..2_000).collect::<Vec<_>>())?;
        let round_trip_operand = |value: &DistinctSketch| {
            let bytes = value
                .try_to_canonical_bytes()
                .expect("valid operand must encode");
            read_fixture(&bytes, value.profile()).expect("encoded operand must decode")
        };
        let a = round_trip_operand(&a);
        let b = round_trip_operand(&b);
        let c = round_trip_operand(&c);

        let mut ab = a.clone();
        ab.try_merge(&b)?;
        let mut ba = b.clone();
        ba.try_merge(&a)?;
        assert_eq!(
            ab.try_to_canonical_bytes()
                .expect("merged state must encode"),
            ba.try_to_canonical_bytes()
                .expect("merged state must encode")
        );

        let mut ab_c = ab;
        ab_c.try_merge(&c)?;
        let mut bc = b;
        bc.try_merge(&c)?;
        let mut a_bc = a.clone();
        a_bc.try_merge(&bc)?;
        assert_eq!(
            ab_c.try_to_canonical_bytes()
                .expect("left-associated merge must encode"),
            a_bc.try_to_canonical_bytes()
                .expect("right-associated merge must encode")
        );
        let direct = populated(&(0_u64..2_000).collect::<Vec<_>>())?;
        assert_eq!(ab_c, direct);

        let before_idempotent = a.try_to_canonical_bytes().expect("valid state must encode");
        let mut idempotent = a.clone();
        idempotent.try_merge(&a)?;
        assert_eq!(
            idempotent
                .try_to_canonical_bytes()
                .expect("idempotent merge must encode"),
            before_idempotent
        );

        let before_delete = ab_c
            .try_to_canonical_bytes()
            .expect("valid state must encode");
        assert_eq!(
            ab_c.try_remove(b"cannot-delete"),
            Err(DistinctError::RebuildRequired)
        );
        assert_eq!(
            ab_c.try_to_canonical_bytes()
                .expect("rejected deletion must preserve encodability"),
            before_delete
        );
        Ok(())
    }

    #[test]
    fn canonical_decoder_rejects_every_truncation_and_trailing_byte() {
        let small_profile = DistinctProfile {
            precision: MIN_PRECISION,
            hash_algorithm: DistinctHashAlgorithm::SeededHasherV1,
            seed: 0x5452_554e_4341_5445,
            max_registers: 1 << MIN_PRECISION,
        };
        let mut value = DistinctSketch::try_new(small_profile).expect("small valid profile");
        value.observe(b"truncation");
        let encoded = value.try_to_canonical_bytes().expect("small valid sketch");

        for cutoff in 0..encoded.len() {
            assert!(
                matches!(
                    read_fixture(&encoded[..cutoff], small_profile),
                    Err(DistinctCodecError::Truncated { .. })
                ),
                "prefix of length {cutoff} was not rejected as truncated"
            );
        }

        let mut trailing = encoded;
        trailing.extend_from_slice(&[0, 1, 2]);
        assert_eq!(
            read_fixture(&trailing, small_profile),
            Err(DistinctCodecError::TrailingBytes {
                offset: CANONICAL_HEADER_BYTES + (1 << MIN_PRECISION),
                remaining: 3,
            })
        );
    }

    #[test]
    fn canonical_decoder_rejects_wrong_identity_shape_bounds_and_ranks() {
        let small_profile = DistinctProfile {
            precision: MIN_PRECISION,
            hash_algorithm: DistinctHashAlgorithm::SeededHasherV1,
            seed: 0x4d41_4c46_4f52_4d45,
            max_registers: 1 << MIN_PRECISION,
        };
        let value = DistinctSketch::try_new(small_profile).expect("small valid profile");
        let encoded = value.try_to_canonical_bytes().expect("small valid sketch");

        let mut wrong_magic = encoded.clone();
        wrong_magic[0] ^= 1;
        assert!(matches!(
            read_fixture(&wrong_magic, small_profile),
            Err(DistinctCodecError::MagicMismatch { .. })
        ));

        let mut wrong_version = encoded.clone();
        wrong_version[VERSION_OFFSET..HASH_ALGORITHM_OFFSET].copy_from_slice(&2_u16.to_be_bytes());
        assert_eq!(
            read_fixture(&wrong_version, small_profile),
            Err(DistinctCodecError::UnsupportedVersion { actual: 2 })
        );

        let mut wrong_hash_algorithm = encoded.clone();
        wrong_hash_algorithm[HASH_ALGORITHM_OFFSET] = 2;
        assert_eq!(
            read_fixture(&wrong_hash_algorithm, small_profile),
            Err(DistinctCodecError::UnsupportedHashAlgorithm { actual: 2 })
        );

        for bad_precision in [MIN_PRECISION - 1, MAX_PRECISION + 1] {
            let mut wrong_precision = encoded.clone();
            wrong_precision[PRECISION_OFFSET] = bad_precision;
            let bad_profile = DistinctProfile {
                precision: bad_precision,
                ..small_profile
            };
            assert_eq!(
                read_fixture(&wrong_precision, bad_profile),
                Err(DistinctCodecError::State(
                    DistinctError::PrecisionOutOfRange {
                        requested: bad_precision,
                        minimum: MIN_PRECISION,
                        maximum: MAX_PRECISION,
                    }
                ))
            );
        }

        let mut insufficient_ceiling = encoded.clone();
        insufficient_ceiling[MAX_REGISTERS_OFFSET..REGISTER_COUNT_OFFSET]
            .copy_from_slice(&((1_u64 << MIN_PRECISION) - 1).to_be_bytes());
        let insufficient_profile = DistinctProfile {
            max_registers: (1 << MIN_PRECISION) - 1,
            ..small_profile
        };
        assert_eq!(
            read_fixture(&insufficient_ceiling, insufficient_profile),
            Err(DistinctCodecError::State(
                DistinctError::RegisterLimitExceeded {
                    requested: 1 << MIN_PRECISION,
                    limit: (1 << MIN_PRECISION) - 1,
                }
            ))
        );

        let mut wrong_count = encoded.clone();
        wrong_count[REGISTER_COUNT_OFFSET..CANONICAL_HEADER_BYTES]
            .copy_from_slice(&((1_u64 << MIN_PRECISION) - 1).to_be_bytes());
        assert_eq!(
            read_fixture(&wrong_count, small_profile),
            Err(DistinctCodecError::RegisterCountMismatch {
                expected: 1 << MIN_PRECISION,
                actual: (1 << MIN_PRECISION) - 1,
            })
        );

        let mut enormous_count = encoded.clone();
        enormous_count[REGISTER_COUNT_OFFSET..CANONICAL_HEADER_BYTES]
            .copy_from_slice(&u64::MAX.to_be_bytes());
        assert!(matches!(
            read_fixture(&enormous_count, small_profile),
            Err(DistinctCodecError::IntegerUnrepresentable)
                | Err(DistinctCodecError::RegisterCountMismatch { .. })
        ));

        let bad_index = 7;
        let bad_rank = maximum_rank(MIN_PRECISION) + 1;
        let mut bad_register = encoded;
        bad_register[CANONICAL_HEADER_BYTES + bad_index] = bad_rank;
        assert_eq!(
            read_fixture(&bad_register, small_profile),
            Err(DistinctCodecError::State(
                DistinctError::NonCanonicalRegisters {
                    index: bad_index,
                    value: bad_rank,
                    maximum: maximum_rank(MIN_PRECISION),
                }
            ))
        );
    }

    #[test]
    fn canonical_encoder_rejects_noncanonical_internal_state() {
        let small_profile = DistinctProfile {
            precision: MIN_PRECISION,
            hash_algorithm: DistinctHashAlgorithm::SeededHasherV1,
            seed: 0x454e_434f_4445_5252,
            max_registers: 1 << MIN_PRECISION,
        };
        let value = DistinctSketch::try_new(small_profile).expect("small valid profile");

        let mut invalid_rank = value.clone();
        invalid_rank.registers[3] = maximum_rank(MIN_PRECISION) + 1;
        assert_eq!(
            invalid_rank.try_to_canonical_bytes(),
            Err(DistinctCodecError::State(
                DistinctError::NonCanonicalRegisters {
                    index: 3,
                    value: maximum_rank(MIN_PRECISION) + 1,
                    maximum: maximum_rank(MIN_PRECISION),
                }
            ))
        );

        let mut missing_register = value;
        missing_register.registers.pop();
        assert_eq!(
            missing_register.try_to_canonical_bytes(),
            Err(DistinctCodecError::State(
                DistinctError::NonCanonicalRegisters {
                    index: (1 << MIN_PRECISION) - 1,
                    value: 0,
                    maximum: maximum_rank(MIN_PRECISION),
                }
            ))
        );
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
                hash_algorithm: DistinctHashAlgorithm::SeededHasherV1,
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
