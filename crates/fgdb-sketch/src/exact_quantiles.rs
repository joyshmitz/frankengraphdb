//! Exact bounded quantile summaries for small statistics windows.
//!
//! This profile stores a canonical sorted multiset until its explicit ceiling.
//! Reaching that ceiling is a typed escalation signal; it never silently swaps
//! to an approximate representation with a different error contract.

use core::fmt;
use std::collections::TryReserveError;

/// Typed exact-summary transition failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExactQuantileError {
    /// Adding observations would exceed the immutable profile ceiling.
    ObservationLimitExceeded {
        /// Attempted completed length.
        attempted: usize,
        /// Immutable profile ceiling.
        maximum: usize,
    },
    /// Length arithmetic exceeded the platform index domain.
    LengthOverflow,
    /// The allocator rejected a checked reservation.
    AllocationFailed {
        /// Values requested for the replacement representation.
        requested: usize,
    },
    /// Merge operands have different ceilings.
    ProfileMismatch {
        /// Receiver ceiling.
        left_maximum: usize,
        /// Other ceiling.
        right_maximum: usize,
    },
    /// The requested value does not occur in the multiset.
    MissingObservation {
        /// Value the caller attempted to remove.
        value: u64,
    },
    /// A rational quantile must satisfy `denominator > 0` and
    /// `numerator <= denominator`.
    InvalidQuantile {
        /// Requested numerator.
        numerator: u64,
        /// Requested denominator.
        denominator: u64,
    },
}

impl fmt::Display for ExactQuantileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::ObservationLimitExceeded { attempted, maximum } => write!(
                formatter,
                "exact quantile summary would contain {attempted} observations, maximum is {maximum}"
            ),
            Self::LengthOverflow => formatter.write_str("exact quantile summary length overflow"),
            Self::AllocationFailed { requested } => write!(
                formatter,
                "could not reserve {requested} exact quantile observations"
            ),
            Self::ProfileMismatch {
                left_maximum,
                right_maximum,
            } => write!(
                formatter,
                "exact quantile ceilings differ: {left_maximum} versus {right_maximum}"
            ),
            Self::MissingObservation { value } => {
                write!(
                    formatter,
                    "exact quantile summary does not contain value {value}"
                )
            }
            Self::InvalidQuantile {
                numerator,
                denominator,
            } => write!(
                formatter,
                "invalid quantile {numerator}/{denominator}; require 0 <= numerator <= denominator and denominator > 0"
            ),
        }
    }
}

impl std::error::Error for ExactQuantileError {}

/// Canonical sorted multiset with exact selection and deletion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactQuantileSketch {
    max_observations: usize,
    values: Vec<u64>,
}

impl ExactQuantileSketch {
    /// Creates an empty, allocation-free exact summary.
    #[must_use]
    pub const fn new(max_observations: usize) -> Self {
        Self {
            max_observations,
            values: Vec::new(),
        }
    }

    /// Creates an empty summary with a checked allocation hint.
    pub fn try_with_capacity(
        max_observations: usize,
        capacity: usize,
    ) -> Result<Self, ExactQuantileError> {
        if capacity > max_observations {
            return Err(ExactQuantileError::ObservationLimitExceeded {
                attempted: capacity,
                maximum: max_observations,
            });
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(capacity)
            .map_err(|_: TryReserveError| ExactQuantileError::AllocationFailed {
                requested: capacity,
            })?;
        Ok(Self {
            max_observations,
            values,
        })
    }

    /// Immutable observation ceiling.
    #[must_use]
    pub const fn max_observations(&self) -> usize {
        self.max_observations
    }

    /// Current multiset cardinality, including duplicates.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the multiset is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Canonical nondecreasing multiset state.
    #[must_use]
    pub fn canonical_values(&self) -> &[u64] {
        &self.values
    }

    /// Inserts one value while preserving canonical order.
    ///
    /// Limit and allocation failures occur before the multiset changes.
    pub fn try_observe(&mut self, value: u64) -> Result<(), ExactQuantileError> {
        let attempted = self
            .values
            .len()
            .checked_add(1)
            .ok_or(ExactQuantileError::LengthOverflow)?;
        if attempted > self.max_observations {
            return Err(ExactQuantileError::ObservationLimitExceeded {
                attempted,
                maximum: self.max_observations,
            });
        }
        self.values.try_reserve(1).map_err(|_: TryReserveError| {
            ExactQuantileError::AllocationFailed {
                requested: attempted,
            }
        })?;
        let insertion = self.values.partition_point(|existing| *existing <= value);
        self.values.insert(insertion, value);
        Ok(())
    }

    /// Removes one occurrence of `value`.
    pub fn try_remove(&mut self, value: u64) -> Result<(), ExactQuantileError> {
        let index = self
            .values
            .binary_search(&value)
            .map_err(|_| ExactQuantileError::MissingObservation { value })?;
        self.values.remove(index);
        Ok(())
    }

    /// Atomically merges another summary with the identical ceiling.
    pub fn try_merge(&mut self, other: &Self) -> Result<(), ExactQuantileError> {
        if self.max_observations != other.max_observations {
            return Err(ExactQuantileError::ProfileMismatch {
                left_maximum: self.max_observations,
                right_maximum: other.max_observations,
            });
        }
        let merged_len = self
            .values
            .len()
            .checked_add(other.values.len())
            .ok_or(ExactQuantileError::LengthOverflow)?;
        if merged_len > self.max_observations {
            return Err(ExactQuantileError::ObservationLimitExceeded {
                attempted: merged_len,
                maximum: self.max_observations,
            });
        }

        let mut merged = Vec::new();
        merged
            .try_reserve_exact(merged_len)
            .map_err(|_: TryReserveError| ExactQuantileError::AllocationFailed {
                requested: merged_len,
            })?;
        let mut left = self.values.iter().copied().peekable();
        let mut right = other.values.iter().copied().peekable();
        while let (Some(left_value), Some(right_value)) = (left.peek(), right.peek()) {
            if left_value <= right_value {
                if let Some(value) = left.next() {
                    merged.push(value);
                }
            } else if let Some(value) = right.next() {
                merged.push(value);
            }
        }
        merged.extend(left);
        merged.extend(right);
        self.values = merged;
        Ok(())
    }

    /// Returns the zero-based ordered observation.
    #[must_use]
    pub fn select(&self, ordinal: usize) -> Option<u64> {
        self.values.get(ordinal).copied()
    }

    /// Returns the deterministic lower rational quantile.
    ///
    /// For `n` observations the selected index is
    /// `floor(numerator * (n - 1) / denominator)`, making `0/denominator` the
    /// minimum and `denominator/denominator` the maximum.
    pub fn quantile(
        &self,
        numerator: u64,
        denominator: u64,
    ) -> Result<Option<u64>, ExactQuantileError> {
        if denominator == 0 || numerator > denominator {
            return Err(ExactQuantileError::InvalidQuantile {
                numerator,
                denominator,
            });
        }
        let Some(last_index) = self.values.len().checked_sub(1) else {
            return Ok(None);
        };
        let scaled = u128::from(numerator) * last_index as u128;
        let index = (scaled / u128::from(denominator)) as usize;
        Ok(self.values.get(index).copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(values: &[u64]) -> ExactQuantileSketch {
        let mut sketch = ExactQuantileSketch::new(100);
        for &value in values {
            sketch.try_observe(value).expect("within ceiling");
        }
        sketch
    }

    #[test]
    fn insertion_order_canonicalizes_the_multiset() {
        let forward = summary(&[9, 1, 5, 5, 2, u64::MAX, 0]);
        let reverse = summary(&[0, u64::MAX, 2, 5, 5, 1, 9]);
        assert_eq!(forward, reverse);
        assert_eq!(forward.canonical_values(), &[0, 1, 2, 5, 5, 9, u64::MAX]);
    }

    #[test]
    fn rational_quantiles_have_fixed_endpoint_and_rounding_semantics() {
        let sketch = summary(&[10, 20, 30, 40, 50]);
        assert_eq!(sketch.quantile(0, 1), Ok(Some(10)));
        assert_eq!(sketch.quantile(1, 4), Ok(Some(20)));
        assert_eq!(sketch.quantile(1, 2), Ok(Some(30)));
        assert_eq!(sketch.quantile(3, 4), Ok(Some(40)));
        assert_eq!(sketch.quantile(1, 1), Ok(Some(50)));
        assert_eq!(
            sketch.quantile(2, 1),
            Err(ExactQuantileError::InvalidQuantile {
                numerator: 2,
                denominator: 1,
            })
        );
        assert_eq!(
            sketch.quantile(0, 0),
            Err(ExactQuantileError::InvalidQuantile {
                numerator: 0,
                denominator: 0,
            })
        );
        assert_eq!(ExactQuantileSketch::new(0).quantile(0, 1), Ok(None));
    }

    #[test]
    fn deletion_is_exact_for_duplicates_and_missing_values_are_typed() {
        let mut sketch = summary(&[1, 2, 2, 2, 3]);
        sketch.try_remove(2).expect("duplicate exists");
        assert_eq!(sketch.canonical_values(), &[1, 2, 2, 3]);
        sketch.try_remove(2).expect("duplicate exists");
        sketch.try_remove(2).expect("duplicate exists");
        assert_eq!(
            sketch.try_remove(2),
            Err(ExactQuantileError::MissingObservation { value: 2 })
        );
        assert_eq!(sketch.canonical_values(), &[1, 3]);
    }

    #[test]
    fn merge_is_commutative_and_associative() {
        let a = summary(&[9, 1, 5]);
        let b = summary(&[4, 4, 8]);
        let c = summary(&[0, 10]);

        let mut left = a.clone();
        left.try_merge(&b).expect("same profile");
        let mut right = b.clone();
        right.try_merge(&a).expect("same profile");
        assert_eq!(left, right);

        let mut ab_c = left;
        ab_c.try_merge(&c).expect("same profile");
        let mut bc = b;
        bc.try_merge(&c).expect("same profile");
        let mut a_bc = a;
        a_bc.try_merge(&bc).expect("same profile");
        assert_eq!(ab_c, a_bc);
    }

    #[test]
    fn limit_and_profile_failures_are_atomic() {
        let mut bounded = ExactQuantileSketch::new(2);
        bounded.try_observe(1).expect("within ceiling");
        bounded.try_observe(2).expect("within ceiling");
        let before = bounded.clone();
        assert_eq!(
            bounded.try_observe(3),
            Err(ExactQuantileError::ObservationLimitExceeded {
                attempted: 3,
                maximum: 2,
            })
        );
        assert_eq!(bounded, before);

        let other = ExactQuantileSketch::new(3);
        assert_eq!(
            bounded.try_merge(&other),
            Err(ExactQuantileError::ProfileMismatch {
                left_maximum: 2,
                right_maximum: 3,
            })
        );
        assert_eq!(bounded, before);
    }
}
