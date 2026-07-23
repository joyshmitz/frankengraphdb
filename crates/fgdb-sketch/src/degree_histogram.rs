//! Mergeable logarithmic histograms for authorized graph degrees.
//!
//! The 65 buckets are fixed and canonical: bucket zero contains degree zero;
//! bucket `b > 0` contains values whose unsigned bit length is `b`. Unlike an
//! insertion-only frequency sketch, this summary supports exact deletion at
//! its declared bucket resolution.

use core::fmt;

/// Number of canonical degree buckets.
pub const DEGREE_BUCKETS: usize = u64::BITS as usize + 1;

/// Inclusive value interval represented by one bucket.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DegreeBucket {
    /// Canonical bucket index in `0..65`.
    pub index: u8,
    /// Smallest degree in this bucket.
    pub lower: u64,
    /// Largest degree in this bucket.
    pub upper: u64,
    /// Number of observations currently in this bucket.
    pub count: u64,
}

/// Typed histogram transition failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DegreeHistogramError {
    /// The configured total-observation ceiling would be exceeded.
    ObservationLimitExceeded {
        /// Configured maximum observation count.
        maximum: u64,
    },
    /// A bucket or total-count addition overflowed.
    CountOverflow,
    /// A deletion named a bucket with no observations.
    MissingObservation {
        /// Degree whose bucket is empty.
        degree: u64,
        /// Canonical empty bucket.
        bucket: u8,
    },
    /// Merge operands use different observation ceilings.
    ProfileMismatch {
        /// Receiver ceiling.
        left_maximum: u64,
        /// Other ceiling.
        right_maximum: u64,
    },
}

impl fmt::Display for DegreeHistogramError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::ObservationLimitExceeded { maximum } => write!(
                formatter,
                "degree histogram reached its {maximum}-observation ceiling"
            ),
            Self::CountOverflow => formatter.write_str("degree histogram count overflow"),
            Self::MissingObservation { degree, bucket } => write!(
                formatter,
                "cannot remove degree {degree}: canonical bucket {bucket} is empty"
            ),
            Self::ProfileMismatch {
                left_maximum,
                right_maximum,
            } => write!(
                formatter,
                "degree histogram ceilings differ: {left_maximum} versus {right_maximum}"
            ),
        }
    }
}

impl std::error::Error for DegreeHistogramError {}

/// Exact-at-bucket-resolution histogram with checked merge and deletion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DegreeHistogram {
    counts: [u64; DEGREE_BUCKETS],
    total: u64,
    max_observations: u64,
}

impl DegreeHistogram {
    /// Creates an empty histogram with an explicit observation ceiling.
    #[must_use]
    pub const fn new(max_observations: u64) -> Self {
        Self {
            counts: [0; DEGREE_BUCKETS],
            total: 0,
            max_observations,
        }
    }

    /// Configured total-observation ceiling.
    #[must_use]
    pub const fn max_observations(&self) -> u64 {
        self.max_observations
    }

    /// Number of observations.
    #[must_use]
    pub const fn len(&self) -> u64 {
        self.total
    }

    /// Whether no degree is represented.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// Canonical bucket counts in increasing degree order.
    #[must_use]
    pub const fn canonical_counts(&self) -> &[u64; DEGREE_BUCKETS] {
        &self.counts
    }

    /// Adds one degree observation.
    pub fn try_observe(&mut self, degree: u64) -> Result<(), DegreeHistogramError> {
        let next_total = self
            .total
            .checked_add(1)
            .ok_or(DegreeHistogramError::CountOverflow)?;
        if next_total > self.max_observations {
            return Err(DegreeHistogramError::ObservationLimitExceeded {
                maximum: self.max_observations,
            });
        }
        let index = bucket_index(degree);
        let next_bucket = self.counts[index]
            .checked_add(1)
            .ok_or(DegreeHistogramError::CountOverflow)?;
        self.counts[index] = next_bucket;
        self.total = next_total;
        Ok(())
    }

    /// Removes one degree observation at the declared bucket resolution.
    pub fn try_remove(&mut self, degree: u64) -> Result<(), DegreeHistogramError> {
        let index = bucket_index(degree);
        if self.counts[index] == 0 {
            return Err(DegreeHistogramError::MissingObservation {
                degree,
                bucket: index as u8,
            });
        }
        self.counts[index] -= 1;
        self.total -= 1;
        Ok(())
    }

    /// Merges another histogram after checking the complete profile and result.
    pub fn try_merge(&mut self, other: &Self) -> Result<(), DegreeHistogramError> {
        if self.max_observations != other.max_observations {
            return Err(DegreeHistogramError::ProfileMismatch {
                left_maximum: self.max_observations,
                right_maximum: other.max_observations,
            });
        }
        let next_total = self
            .total
            .checked_add(other.total)
            .ok_or(DegreeHistogramError::CountOverflow)?;
        if next_total > self.max_observations {
            return Err(DegreeHistogramError::ObservationLimitExceeded {
                maximum: self.max_observations,
            });
        }
        for (&left, &right) in self.counts.iter().zip(&other.counts) {
            left.checked_add(right)
                .ok_or(DegreeHistogramError::CountOverflow)?;
        }
        for (left, &right) in self.counts.iter_mut().zip(&other.counts) {
            *left += right;
        }
        self.total = next_total;
        Ok(())
    }

    /// Returns the bucket containing the zero-based ordered observation.
    ///
    /// The result is an interval because values inside one logarithmic bucket
    /// are intentionally indistinguishable.
    #[must_use]
    pub fn select_bucket(&self, ordinal: u64) -> Option<DegreeBucket> {
        if ordinal >= self.total {
            return None;
        }
        let mut preceding = 0_u64;
        for (index, &count) in self.counts.iter().enumerate() {
            let after = preceding + count;
            if ordinal < after {
                let (lower, upper) = bucket_bounds(index);
                return Some(DegreeBucket {
                    index: index as u8,
                    lower,
                    upper,
                    count,
                });
            }
            preceding = after;
        }
        None
    }

    /// Returns one bucket's canonical interval and count.
    #[must_use]
    pub fn bucket(&self, index: u8) -> Option<DegreeBucket> {
        let index = usize::from(index);
        let &count = self.counts.get(index)?;
        let (lower, upper) = bucket_bounds(index);
        Some(DegreeBucket {
            index: index as u8,
            lower,
            upper,
            count,
        })
    }
}

fn bucket_index(degree: u64) -> usize {
    if degree == 0 {
        0
    } else {
        (u64::BITS - degree.leading_zeros()) as usize
    }
}

fn bucket_bounds(index: usize) -> (u64, u64) {
    debug_assert!(index < DEGREE_BUCKETS);
    match index {
        0 => (0, 0),
        64 => (1_u64 << 63, u64::MAX),
        _ => {
            let lower = 1_u64 << (index - 1);
            (lower, (lower << 1) - 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_power_boundary_maps_to_its_canonical_interval() {
        let fixtures = [
            (0, 0, 0, 0),
            (1, 1, 1, 1),
            (2, 2, 2, 3),
            (3, 2, 2, 3),
            (4, 3, 4, 7),
            (7, 3, 4, 7),
            (8, 4, 8, 15),
            (u64::MAX, 64, 1_u64 << 63, u64::MAX),
        ];
        for (degree, index, lower, upper) in fixtures {
            assert_eq!(bucket_index(degree), index);
            assert_eq!(bucket_bounds(index), (lower, upper));
        }
    }

    #[test]
    fn observe_remove_and_selection_preserve_bucket_semantics() {
        let mut histogram = DegreeHistogram::new(20);
        for degree in [0, 1, 2, 3, 3, 8, u64::MAX] {
            histogram.try_observe(degree).expect("within ceiling");
        }
        assert_eq!(histogram.len(), 7);
        assert_eq!(
            histogram.select_bucket(0),
            Some(DegreeBucket {
                index: 0,
                lower: 0,
                upper: 0,
                count: 1,
            })
        );
        assert_eq!(
            histogram.select_bucket(3).map(|bucket| bucket.index),
            Some(2)
        );
        assert_eq!(
            histogram.select_bucket(6).map(|bucket| bucket.index),
            Some(64)
        );
        assert_eq!(histogram.select_bucket(7), None);

        histogram.try_remove(2).expect("bucket has observations");
        assert_eq!(histogram.bucket(2).map(|bucket| bucket.count), Some(2));
        histogram.try_remove(3).expect("bucket has observations");
        histogram.try_remove(3).expect("bucket has observations");
        assert_eq!(
            histogram.try_remove(2),
            Err(DegreeHistogramError::MissingObservation {
                degree: 2,
                bucket: 2,
            })
        );
    }

    #[test]
    fn merge_is_checked_commutative_and_associative() {
        fn part(values: &[u64]) -> DegreeHistogram {
            let mut histogram = DegreeHistogram::new(100);
            for &value in values {
                histogram.try_observe(value).expect("within ceiling");
            }
            histogram
        }
        let a = part(&[0, 1, 3]);
        let b = part(&[2, 8, 9]);
        let c = part(&[u64::MAX, 16]);

        let mut left = a.clone();
        left.try_merge(&b).expect("matching profile");
        let mut right = b.clone();
        right.try_merge(&a).expect("matching profile");
        assert_eq!(left, right);

        let mut ab_c = left;
        ab_c.try_merge(&c).expect("matching profile");
        let mut bc = b;
        bc.try_merge(&c).expect("matching profile");
        let mut a_bc = a;
        a_bc.try_merge(&bc).expect("matching profile");
        assert_eq!(ab_c, a_bc);
    }

    #[test]
    fn profile_and_count_failures_are_atomic() {
        let mut histogram = DegreeHistogram::new(1);
        histogram.try_observe(7).expect("first observation");
        let before = histogram.clone();
        assert_eq!(
            histogram.try_observe(9),
            Err(DegreeHistogramError::ObservationLimitExceeded { maximum: 1 })
        );
        assert_eq!(histogram, before);

        let other = DegreeHistogram::new(2);
        assert_eq!(
            histogram.try_merge(&other),
            Err(DegreeHistogramError::ProfileMismatch {
                left_maximum: 1,
                right_maximum: 2,
            })
        );
        assert_eq!(histogram, before);
    }
}
