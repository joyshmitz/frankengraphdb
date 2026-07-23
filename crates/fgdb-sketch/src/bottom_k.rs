//! Deterministic bottom-k sampling over canonical byte observations.
//!
//! This sketch has distinct-set semantics: observing the same byte string more
//! than once is idempotent. Samples are ordered by `(stable_hash, bytes)`, so a
//! hash collision cannot make the logical state ambiguous. The retained vector
//! is always sorted, deduplicated, and no longer than the profile's `k`.
//!
//! Deletion is deliberately conservative. While the sample is not saturated,
//! it contains the complete distinct universe and a retained observation can be
//! removed exactly. Once saturated, removing a retained observation would
//! require the unknown `(k + 1)`th candidate and therefore returns
//! [`BottomKError::RebuildRequired`] without changing state. Removing an
//! unretained observation is always an exact no-op: in a saturated sketch it
//! cannot affect the current bottom-k, and in an unsaturated sketch its absence
//! proves it was never part of the represented distinct set.

#![forbid(unsafe_code)]

use core::cmp::Ordering;
use core::fmt;
use core::hash::Hasher;
use fgdb_collections::hash_table::SeededHasher;

const HASH_DOMAIN: &[u8] = b"fgdb:bottom-k:observation:v1";

/// Complete behavior, hash, and resource profile for a bottom-k sketch.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BottomKProfile {
    /// Maximum number of distinct samples retained.
    pub k: usize,
    /// Explicit deterministic seed for the stable hash.
    pub seed: u64,
    /// Maximum accepted length of one canonical observation.
    pub max_observation_bytes: usize,
    /// Maximum sum of observation bytes retained in the sample.
    pub max_sample_bytes: usize,
}

impl BottomKProfile {
    /// Creates a complete profile.
    #[must_use]
    pub const fn new(
        k: usize,
        seed: u64,
        max_observation_bytes: usize,
        max_sample_bytes: usize,
    ) -> Self {
        Self {
            k,
            seed,
            max_observation_bytes,
            max_sample_bytes,
        }
    }
}

/// Allocation owned by a failed bottom-k transition.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BottomKAllocationTarget {
    /// Sorted sample directory.
    SampleDirectory,
    /// Bytes for one retained observation.
    ObservationBytes,
    /// Temporary sorted directory for an atomic merge.
    MergeDirectory,
    /// Observation bytes cloned into an atomic merge result.
    MergeObservationBytes,
}

/// Typed construction or transition failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BottomKError {
    /// A zero-sized sample has no useful bottom-k semantics.
    EmptySampleSize,
    /// One observation exceeds the complete profile's byte ceiling.
    ObservationTooLarge {
        /// Rejected observation length.
        bytes: usize,
        /// Profile ceiling.
        maximum: usize,
    },
    /// Retaining a candidate would exceed the sample payload ceiling.
    SampleByteLimitExceeded {
        /// Payload bytes the transition would retain.
        requested: usize,
        /// Profile ceiling.
        maximum: usize,
    },
    /// Checked payload-size arithmetic overflowed.
    SampleByteCountOverflow,
    /// The allocator rejected a checked reservation.
    AllocationFailed {
        /// Component whose reservation failed.
        target: BottomKAllocationTarget,
        /// Exact entries or bytes requested.
        requested: usize,
    },
    /// Merge operands do not use the identical complete profile.
    ProfileMismatch,
    /// Exact deletion requires candidates that bottom-k does not retain.
    RebuildRequired {
        /// Stable hash of the retained observation whose removal was refused.
        observation_hash: u64,
    },
}

impl fmt::Display for BottomKError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::EmptySampleSize => formatter.write_str("bottom-k sample size must be nonzero"),
            Self::ObservationTooLarge { bytes, maximum } => write!(
                formatter,
                "bottom-k observation has {bytes} bytes, maximum is {maximum}"
            ),
            Self::SampleByteLimitExceeded { requested, maximum } => write!(
                formatter,
                "bottom-k sample would retain {requested} observation bytes, maximum is {maximum}"
            ),
            Self::SampleByteCountOverflow => {
                formatter.write_str("bottom-k sample byte count overflowed")
            }
            Self::AllocationFailed { target, requested } => write!(
                formatter,
                "could not reserve {requested} units for bottom-k {target:?}"
            ),
            Self::ProfileMismatch => {
                formatter.write_str("cannot merge bottom-k sketches with different profiles")
            }
            Self::RebuildRequired { observation_hash } => write!(
                formatter,
                "removing retained bottom-k observation {observation_hash:#018x} requires rebuild"
            ),
        }
    }
}

impl std::error::Error for BottomKError {}

/// One canonical retained sample.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BottomKSample {
    hash: u64,
    observation: Vec<u8>,
}

impl BottomKSample {
    /// Seeded stable hash used for sample ordering.
    #[must_use]
    pub const fn hash(&self) -> u64 {
        self.hash
    }

    /// Canonical observation bytes used to break hash collisions.
    #[must_use]
    pub fn observation(&self) -> &[u8] {
        &self.observation
    }
}

impl Ord for BottomKSample {
    fn cmp(&self, other: &Self) -> Ordering {
        self.hash
            .cmp(&other.hash)
            .then_with(|| self.observation.cmp(&other.observation))
    }
}

impl PartialOrd for BottomKSample {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Borrowed canonical logical state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BottomKState<'sketch> {
    /// Complete behavior and resource profile.
    pub profile: BottomKProfile,
    /// Sum of retained canonical observation lengths.
    pub sample_bytes: usize,
    /// Samples sorted by `(hash, observation bytes)` with no duplicates.
    pub samples: &'sketch [BottomKSample],
}

/// Mergeable deterministic distinct bottom-k sample.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BottomKSketch {
    profile: BottomKProfile,
    samples: Vec<BottomKSample>,
    sample_bytes: usize,
}

impl BottomKSketch {
    /// Creates an empty sketch after fallibly reserving its bounded directory.
    pub fn try_new(profile: BottomKProfile) -> Result<Self, BottomKError> {
        if profile.k == 0 {
            return Err(BottomKError::EmptySampleSize);
        }
        let mut samples = Vec::new();
        reserve_samples(
            &mut samples,
            profile.k,
            BottomKAllocationTarget::SampleDirectory,
        )?;
        Ok(Self {
            profile,
            samples,
            sample_bytes: 0,
        })
    }

    /// Complete immutable profile.
    #[must_use]
    pub const fn profile(&self) -> BottomKProfile {
        self.profile
    }

    /// Number of retained distinct samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether no observation is retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Whether the sample has reached `k` and may hide larger candidates.
    #[must_use]
    pub fn is_saturated(&self) -> bool {
        self.samples.len() == self.profile.k
    }

    /// Sum of retained canonical observation lengths.
    #[must_use]
    pub const fn sample_bytes(&self) -> usize {
        self.sample_bytes
    }

    /// Canonical sorted and deduplicated logical state.
    #[must_use]
    pub fn canonical_state(&self) -> BottomKState<'_> {
        BottomKState {
            profile: self.profile,
            sample_bytes: self.sample_bytes,
            samples: &self.samples,
        }
    }

    /// Largest retained sample, or `None` while empty.
    #[must_use]
    pub fn threshold(&self) -> Option<&BottomKSample> {
        self.samples.last()
    }

    /// Computes this profile's stable hash for a bounded observation.
    pub fn try_hash(&self, observation: &[u8]) -> Result<u64, BottomKError> {
        self.validate_observation(observation)?;
        Ok(stable_hash(self.profile.seed, observation))
    }

    /// Returns whether the canonical observation is currently retained.
    pub fn try_contains(&self, observation: &[u8]) -> Result<bool, BottomKError> {
        let hash = self.try_hash(observation)?;
        Ok(self.find(hash, observation).is_ok())
    }

    /// Observes one canonical byte string with distinct-set semantics.
    ///
    /// Duplicate observations and candidates above a saturated threshold are
    /// exact no-ops. Every limit and allocation needed for a retained candidate
    /// is checked before the state changes.
    pub fn try_observe(&mut self, observation: &[u8]) -> Result<(), BottomKError> {
        self.validate_observation(observation)?;
        let hash = stable_hash(self.profile.seed, observation);
        let insertion_index = match self.find(hash, observation) {
            Ok(_) => return Ok(()),
            Err(index) => index,
        };

        if self.is_saturated() && insertion_index == self.profile.k {
            return Ok(());
        }

        let evicted_bytes = if self.is_saturated() {
            self.samples
                .last()
                .map_or(0, |sample| sample.observation.len())
        } else {
            0
        };
        let next_sample_bytes = self
            .sample_bytes
            .checked_sub(evicted_bytes)
            .and_then(|bytes| bytes.checked_add(observation.len()))
            .ok_or(BottomKError::SampleByteCountOverflow)?;
        if next_sample_bytes > self.profile.max_sample_bytes {
            return Err(BottomKError::SampleByteLimitExceeded {
                requested: next_sample_bytes,
                maximum: self.profile.max_sample_bytes,
            });
        }

        if !self.is_saturated() && self.samples.len() == self.samples.capacity() {
            self.samples
                .try_reserve_exact(1)
                .map_err(|_| BottomKError::AllocationFailed {
                    target: BottomKAllocationTarget::SampleDirectory,
                    requested: 1,
                })?;
        }
        let sample =
            try_clone_sample(hash, observation, BottomKAllocationTarget::ObservationBytes)?;

        if self.is_saturated() {
            self.samples.pop();
        }
        self.samples.insert(insertion_index, sample);
        self.sample_bytes = next_sample_bytes;
        Ok(())
    }

    /// Merges the exact bottom-k union of an identical-profile sketch.
    ///
    /// The complete result is built in private fallible storage before the
    /// receiver changes, making profile, resource, and allocation failures
    /// atomic. The algebra is commutative, associative, and idempotent over
    /// successful identical-profile merges.
    pub fn try_merge(&mut self, other: &Self) -> Result<(), BottomKError> {
        if self.profile != other.profile {
            return Err(BottomKError::ProfileMismatch);
        }

        let mut merged = Vec::new();
        reserve_samples(
            &mut merged,
            self.profile.k,
            BottomKAllocationTarget::MergeDirectory,
        )?;
        let mut merged_bytes = 0_usize;
        let mut left = 0_usize;
        let mut right = 0_usize;

        while merged.len() < self.profile.k
            && (left < self.samples.len() || right < other.samples.len())
        {
            let selected = match (self.samples.get(left), other.samples.get(right)) {
                (Some(left_sample), Some(right_sample)) => match left_sample.cmp(right_sample) {
                    Ordering::Less => {
                        left += 1;
                        left_sample
                    }
                    Ordering::Greater => {
                        right += 1;
                        right_sample
                    }
                    Ordering::Equal => {
                        left += 1;
                        right += 1;
                        left_sample
                    }
                },
                (Some(left_sample), None) => {
                    left += 1;
                    left_sample
                }
                (None, Some(right_sample)) => {
                    right += 1;
                    right_sample
                }
                (None, None) => break,
            };

            let next_bytes = merged_bytes
                .checked_add(selected.observation.len())
                .ok_or(BottomKError::SampleByteCountOverflow)?;
            if next_bytes > self.profile.max_sample_bytes {
                return Err(BottomKError::SampleByteLimitExceeded {
                    requested: next_bytes,
                    maximum: self.profile.max_sample_bytes,
                });
            }
            let cloned = try_clone_sample(
                selected.hash,
                &selected.observation,
                BottomKAllocationTarget::MergeObservationBytes,
            )?;
            merged.push(cloned);
            merged_bytes = next_bytes;
        }

        self.samples = merged;
        self.sample_bytes = merged_bytes;
        Ok(())
    }

    /// Removes one distinct observation when bottom-k has enough information.
    ///
    /// See the module-level deletion contract. A rebuild-required result never
    /// mutates the sketch.
    pub fn try_remove(&mut self, observation: &[u8]) -> Result<(), BottomKError> {
        self.validate_observation(observation)?;
        let hash = stable_hash(self.profile.seed, observation);
        let Ok(index) = self.find(hash, observation) else {
            return Ok(());
        };
        if self.is_saturated() {
            return Err(BottomKError::RebuildRequired {
                observation_hash: hash,
            });
        }

        let removed_bytes = self.samples[index].observation.len();
        self.samples.remove(index);
        self.sample_bytes -= removed_bytes;
        Ok(())
    }

    fn validate_observation(&self, observation: &[u8]) -> Result<(), BottomKError> {
        if observation.len() > self.profile.max_observation_bytes {
            return Err(BottomKError::ObservationTooLarge {
                bytes: observation.len(),
                maximum: self.profile.max_observation_bytes,
            });
        }
        Ok(())
    }

    fn find(&self, hash: u64, observation: &[u8]) -> Result<usize, usize> {
        self.samples.binary_search_by(|sample| {
            sample
                .hash
                .cmp(&hash)
                .then_with(|| sample.observation.as_slice().cmp(observation))
        })
    }
}

fn stable_hash(seed: u64, observation: &[u8]) -> u64 {
    let mut hasher = SeededHasher::new(seed);
    hasher.write(HASH_DOMAIN);
    hasher.write_u64(observation.len() as u64);
    hasher.write(observation);
    hasher.finish()
}

fn try_clone_sample(
    hash: u64,
    observation: &[u8],
    target: BottomKAllocationTarget,
) -> Result<BottomKSample, BottomKError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(observation.len())
        .map_err(|_| BottomKError::AllocationFailed {
            target,
            requested: observation.len(),
        })?;
    bytes.extend_from_slice(observation);
    Ok(BottomKSample {
        hash,
        observation: bytes,
    })
}

fn reserve_samples(
    samples: &mut Vec<BottomKSample>,
    requested: usize,
    target: BottomKAllocationTarget,
) -> Result<(), BottomKError> {
    samples
        .try_reserve_exact(requested)
        .map_err(|_| BottomKError::AllocationFailed { target, requested })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(k: usize) -> BottomKProfile {
        BottomKProfile::new(k, 0x424f_5454_4f4d_4b31, 64, 256)
    }

    fn sketch(k: usize) -> BottomKSketch {
        BottomKSketch::try_new(profile(k)).expect("bounded profile")
    }

    fn observe_all(sketch: &mut BottomKSketch, observations: &[&[u8]]) {
        for &observation in observations {
            sketch
                .try_observe(observation)
                .expect("bounded observation");
        }
    }

    #[test]
    fn stable_hash_and_order_are_deterministic() {
        let observations = [
            &b"alpha"[..],
            &b"beta"[..],
            &b"gamma"[..],
            &b"delta"[..],
            &b"epsilon"[..],
        ];
        let mut forward = sketch(3);
        let mut reverse = sketch(3);
        observe_all(&mut forward, &observations);
        for &observation in observations.iter().rev() {
            reverse
                .try_observe(observation)
                .expect("bounded observation");
        }

        assert_eq!(forward, reverse);
        assert!(
            forward
                .canonical_state()
                .samples
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
        assert_eq!(
            stable_hash(profile(3).seed, b"alpha"),
            0x36f6_5c36_d3c8_d99c
        );
        assert_ne!(
            stable_hash(profile(3).seed, b"alpha"),
            stable_hash(profile(3).seed ^ 1, b"alpha")
        );
    }

    #[test]
    fn duplicate_observations_are_idempotent() {
        let mut value = sketch(4);
        value.try_observe(b"same").expect("first observation");
        let before = value.clone();
        for _ in 0..20 {
            value.try_observe(b"same").expect("duplicate observation");
        }
        assert_eq!(value, before);
        assert_eq!(value.len(), 1);
        assert_eq!(value.sample_bytes(), 4);
    }

    #[test]
    fn merge_is_commutative_associative_and_idempotent() {
        fn part(entries: &[&[u8]]) -> BottomKSketch {
            let mut value = sketch(5);
            observe_all(&mut value, entries);
            value
        }

        let a = part(&[b"a", b"d", b"g", b"shared"]);
        let b = part(&[b"b", b"e", b"h", b"shared"]);
        let c = part(&[b"c", b"f", b"i", b"shared"]);

        let mut ab = a.clone();
        ab.try_merge(&b).expect("matching profile");
        let mut ba = b.clone();
        ba.try_merge(&a).expect("matching profile");
        assert_eq!(ab, ba);

        let mut ab_c = ab;
        ab_c.try_merge(&c).expect("matching profile");
        let mut bc = b;
        bc.try_merge(&c).expect("matching profile");
        let mut a_bc = a.clone();
        a_bc.try_merge(&bc).expect("matching profile");
        assert_eq!(ab_c, a_bc);

        let before = a.clone();
        let mut idempotent = a;
        idempotent
            .try_merge(&before)
            .expect("identical profile and state");
        assert_eq!(idempotent, before);
    }

    #[test]
    fn merged_sample_equals_naive_sorted_distinct_union() {
        let left_values = [&b"one"[..], &b"two"[..], &b"three"[..], &b"shared"[..]];
        let right_values = [&b"four"[..], &b"five"[..], &b"six"[..], &b"shared"[..]];
        let mut left = sketch(4);
        let mut right = sketch(4);
        observe_all(&mut left, &left_values);
        observe_all(&mut right, &right_values);
        left.try_merge(&right).expect("matching profile");

        let mut expected = left_values
            .into_iter()
            .chain(right_values)
            .map(|bytes| (stable_hash(profile(4).seed, bytes), bytes))
            .collect::<Vec<_>>();
        expected.sort();
        expected.dedup();
        expected.truncate(4);
        let actual = left
            .canonical_state()
            .samples
            .iter()
            .map(|sample| (sample.hash(), sample.observation()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn construction_observation_and_merge_limits_are_atomic() {
        assert_eq!(
            BottomKSketch::try_new(profile(0)),
            Err(BottomKError::EmptySampleSize)
        );

        let constrained_profile = BottomKProfile::new(2, 7, 3, 5);
        let mut constrained = BottomKSketch::try_new(constrained_profile).expect("bounded profile");
        assert_eq!(
            constrained.try_observe(b"four"),
            Err(BottomKError::ObservationTooLarge {
                bytes: 4,
                maximum: 3,
            })
        );
        constrained.try_observe(b"aaa").expect("first sample fits");
        let before = constrained.clone();
        assert_eq!(
            constrained.try_observe(b"bbb"),
            Err(BottomKError::SampleByteLimitExceeded {
                requested: 6,
                maximum: 5,
            })
        );
        assert_eq!(constrained, before);

        let mut left = BottomKSketch::try_new(constrained_profile).expect("bounded profile");
        let mut right = BottomKSketch::try_new(constrained_profile).expect("bounded profile");
        left.try_observe(b"aaa").expect("individual sample fits");
        right.try_observe(b"bbb").expect("individual sample fits");
        let before = left.clone();
        assert_eq!(
            left.try_merge(&right),
            Err(BottomKError::SampleByteLimitExceeded {
                requested: 6,
                maximum: 5,
            })
        );
        assert_eq!(left, before);

        let different = BottomKSketch::try_new(BottomKProfile {
            seed: constrained_profile.seed ^ 1,
            ..constrained_profile
        })
        .expect("bounded profile");
        assert_eq!(
            left.try_merge(&different),
            Err(BottomKError::ProfileMismatch)
        );
        assert_eq!(left, before);
    }

    #[test]
    fn deletion_is_exact_when_complete_and_rebuilds_when_saturated() {
        let mut complete = sketch(4);
        observe_all(&mut complete, &[b"a", b"b", b"c"]);
        assert_eq!(complete.try_remove(b"absent"), Ok(()));
        complete.try_remove(b"b").expect("complete set can delete");
        assert_eq!(complete.try_contains(b"b"), Ok(false));
        assert_eq!(complete.len(), 2);

        let mut saturated = sketch(2);
        let candidates = [&b"a"[..], &b"b"[..], &b"c"[..], &b"d"[..], &b"e"[..]];
        observe_all(&mut saturated, &candidates);
        let retained = saturated
            .canonical_state()
            .samples
            .iter()
            .map(|sample| sample.observation().to_vec())
            .collect::<Vec<_>>();
        let retained_observation = retained[0].clone();
        let unretained = candidates
            .iter()
            .find(|candidate| !retained.iter().any(|sample| sample == *candidate))
            .copied()
            .expect("more candidates than retained samples");

        let before = saturated.clone();
        assert_eq!(
            saturated.try_remove(&retained_observation),
            Err(BottomKError::RebuildRequired {
                observation_hash: stable_hash(profile(2).seed, &retained_observation),
            })
        );
        assert_eq!(saturated, before);
        assert_eq!(saturated.try_remove(unretained), Ok(()));
        assert_eq!(saturated, before);
    }
}
