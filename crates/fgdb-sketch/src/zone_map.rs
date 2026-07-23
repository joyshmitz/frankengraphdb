//! Canonical bounded byte zone maps.
//!
//! A zone map retains only the lexicographic extrema and observation count of
//! a byte-valued zone. It is an advisory pruning summary, not an exact set:
//! [`ByteZoneMap::may_contain`] can rule values out but cannot prove that an
//! interior value exists.

use core::fmt;
use std::collections::TryReserveError;

/// Complete resource and merge profile for a byte zone map.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ZoneMapProfile {
    /// Maximum bytes accepted in one observed value.
    pub max_value_bytes: usize,
    /// Maximum number of observations represented by one map.
    pub max_observations: u64,
}

/// Endpoint whose exact post-delete replacement is unknowable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ZoneMapEndpoint {
    Minimum,
    Maximum,
}

/// Endpoint allocation named by a fallible copy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ZoneMapAllocation {
    Minimum,
    Maximum,
}

/// Typed construction or state-transition failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ZoneMapError {
    /// An observed value exceeds the profile's byte ceiling.
    ValueLimitExceeded { actual: usize, limit: usize },
    /// An observation or merge exceeds the profile's count ceiling.
    ObservationLimitExceeded { maximum: u64 },
    /// Adding observation counts overflowed `u64`.
    CountOverflow,
    /// The allocator rejected a checked endpoint reservation.
    AllocationFailed {
        endpoint: ZoneMapAllocation,
        requested_bytes: usize,
    },
    /// Merge operands use different complete profiles.
    ProfileMismatch {
        left: ZoneMapProfile,
        right: ZoneMapProfile,
    },
    /// No represented observation can equal the requested value.
    MissingObservation,
    /// Deleting an extremum would require information the summary does not
    /// retain. The map remains unchanged and must be rebuilt from source data.
    RebuildRequired {
        endpoint: ZoneMapEndpoint,
        count: u64,
    },
    /// Private count/extrema state was internally inconsistent.
    InvariantViolation,
}

impl fmt::Display for ZoneMapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::ValueLimitExceeded { actual, limit } => write!(
                formatter,
                "zone-map value has {actual} bytes, configured limit is {limit}"
            ),
            Self::ObservationLimitExceeded { maximum } => write!(
                formatter,
                "zone map reached its {maximum}-observation ceiling"
            ),
            Self::CountOverflow => formatter.write_str("zone-map observation count overflow"),
            Self::AllocationFailed {
                endpoint,
                requested_bytes,
            } => write!(
                formatter,
                "could not reserve {requested_bytes} bytes for zone-map {endpoint:?}"
            ),
            Self::ProfileMismatch { left, right } => write!(
                formatter,
                "cannot merge zone maps with profiles {left:?} and {right:?}"
            ),
            Self::MissingObservation => {
                formatter.write_str("zone map cannot contain the observation being deleted")
            }
            Self::RebuildRequired { endpoint, count } => write!(
                formatter,
                "deleting zone-map {endpoint:?} from {count} observations requires a rebuild"
            ),
            Self::InvariantViolation => {
                formatter.write_str("zone-map count/extrema invariant is inconsistent")
            }
        }
    }
}

impl std::error::Error for ZoneMapError {}

/// Borrowed canonical logical state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ZoneMapState<'map> {
    pub profile: ZoneMapProfile,
    pub count: u64,
    pub minimum: Option<&'map [u8]>,
    pub maximum: Option<&'map [u8]>,
}

/// Mergeable lexicographic extrema over bounded owned byte values.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ByteZoneMap {
    profile: ZoneMapProfile,
    count: u64,
    minimum: Option<Vec<u8>>,
    maximum: Option<Vec<u8>>,
}

impl ByteZoneMap {
    /// Creates an empty map without allocating.
    #[must_use]
    pub const fn new(profile: ZoneMapProfile) -> Self {
        Self {
            profile,
            count: 0,
            minimum: None,
            maximum: None,
        }
    }

    /// Returns the complete merge and resource profile.
    #[must_use]
    pub const fn profile(&self) -> ZoneMapProfile {
        self.profile
    }

    /// Returns the represented observation count.
    #[must_use]
    pub const fn len(&self) -> u64 {
        self.count
    }

    /// Returns whether no observations are represented.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns the lexicographically least observed bytes.
    #[must_use]
    pub fn minimum(&self) -> Option<&[u8]> {
        self.minimum.as_deref()
    }

    /// Returns the lexicographically greatest observed bytes.
    #[must_use]
    pub fn maximum(&self) -> Option<&[u8]> {
        self.maximum.as_deref()
    }

    /// Returns the canonical logical state.
    #[must_use]
    pub fn canonical_state(&self) -> ZoneMapState<'_> {
        ZoneMapState {
            profile: self.profile,
            count: self.count,
            minimum: self.minimum(),
            maximum: self.maximum(),
        }
    }

    /// Adds one observation after preflighting every allocation and count
    /// transition. Failure leaves the map byte-for-byte unchanged.
    pub fn try_observe(&mut self, value: &[u8]) -> Result<(), ZoneMapError> {
        self.validate_value(value)?;
        let next_count = self.checked_combined_count(1)?;

        if self.count == 0 {
            let minimum = try_copy_endpoint(value, ZoneMapAllocation::Minimum)?;
            let maximum = try_copy_endpoint(value, ZoneMapAllocation::Maximum)?;
            self.minimum = Some(minimum);
            self.maximum = Some(maximum);
            self.count = next_count;
            return Ok(());
        }

        let (Some(minimum), Some(maximum)) = (&self.minimum, &self.maximum) else {
            return Err(ZoneMapError::InvariantViolation);
        };
        let next_minimum = (value < minimum.as_slice())
            .then(|| try_copy_endpoint(value, ZoneMapAllocation::Minimum))
            .transpose()?;
        let next_maximum = (value > maximum.as_slice())
            .then(|| try_copy_endpoint(value, ZoneMapAllocation::Maximum))
            .transpose()?;

        if let Some(minimum) = next_minimum {
            self.minimum = Some(minimum);
        }
        if let Some(maximum) = next_maximum {
            self.maximum = Some(maximum);
        }
        self.count = next_count;
        Ok(())
    }

    /// Merges another map with the identical complete profile.
    ///
    /// Both replacement extrema and the resulting count are prepared before
    /// mutation, making profile, limit, and allocation failures atomic.
    pub fn try_merge(&mut self, other: &Self) -> Result<(), ZoneMapError> {
        if self.profile != other.profile {
            return Err(ZoneMapError::ProfileMismatch {
                left: self.profile,
                right: other.profile,
            });
        }
        if other.count == 0 {
            return Ok(());
        }
        let next_count = self.checked_combined_count(other.count)?;
        let (Some(other_minimum), Some(other_maximum)) = (&other.minimum, &other.maximum) else {
            return Err(ZoneMapError::InvariantViolation);
        };

        let replace_minimum = self
            .minimum
            .as_deref()
            .is_none_or(|minimum| other_minimum.as_slice() < minimum);
        let replace_maximum = self
            .maximum
            .as_deref()
            .is_none_or(|maximum| other_maximum.as_slice() > maximum);
        let next_minimum = replace_minimum
            .then(|| try_copy_endpoint(other_minimum, ZoneMapAllocation::Minimum))
            .transpose()?;
        let next_maximum = replace_maximum
            .then(|| try_copy_endpoint(other_maximum, ZoneMapAllocation::Maximum))
            .transpose()?;

        if let Some(minimum) = next_minimum {
            self.minimum = Some(minimum);
        }
        if let Some(maximum) = next_maximum {
            self.maximum = Some(maximum);
        }
        self.count = next_count;
        Ok(())
    }

    /// Removes one source-confirmed observation when its extrema transition is
    /// exactly derivable from the retained state.
    ///
    /// Interior values never change extrema. Equal extrema prove that every
    /// represented value is identical. Two distinct extrema with count two
    /// prove that each endpoint occurs exactly once. Any other endpoint delete
    /// is rejected as [`ZoneMapError::RebuildRequired`] without mutation.
    pub fn try_remove(&mut self, value: &[u8]) -> Result<(), ZoneMapError> {
        self.validate_value(value)?;
        let (Some(minimum), Some(maximum)) = (&self.minimum, &self.maximum) else {
            return if self.count == 0 {
                Err(ZoneMapError::MissingObservation)
            } else {
                Err(ZoneMapError::InvariantViolation)
            };
        };
        if self.count == 0 {
            return Err(ZoneMapError::InvariantViolation);
        }
        if value < minimum.as_slice() || value > maximum.as_slice() {
            return Err(ZoneMapError::MissingObservation);
        }

        if minimum == maximum {
            if value != minimum.as_slice() {
                return Err(ZoneMapError::MissingObservation);
            }
            self.count -= 1;
            if self.count == 0 {
                self.minimum = None;
                self.maximum = None;
            }
            return Ok(());
        }

        if value == minimum.as_slice() {
            if self.count != 2 {
                return Err(ZoneMapError::RebuildRequired {
                    endpoint: ZoneMapEndpoint::Minimum,
                    count: self.count,
                });
            }
            let replacement = try_copy_endpoint(maximum, ZoneMapAllocation::Minimum)?;
            self.minimum = Some(replacement);
            self.count = 1;
            return Ok(());
        }
        if value == maximum.as_slice() {
            if self.count != 2 {
                return Err(ZoneMapError::RebuildRequired {
                    endpoint: ZoneMapEndpoint::Maximum,
                    count: self.count,
                });
            }
            let replacement = try_copy_endpoint(minimum, ZoneMapAllocation::Maximum)?;
            self.maximum = Some(replacement);
            self.count = 1;
            return Ok(());
        }

        if self.count <= 2 {
            return Err(ZoneMapError::MissingObservation);
        }
        self.count -= 1;
        Ok(())
    }

    /// Returns whether the retained interval permits `value`.
    ///
    /// `false` is definitive; `true` means only that the value lies between
    /// the extrema.
    #[must_use]
    pub fn may_contain(&self, value: &[u8]) -> bool {
        self.minimum
            .as_deref()
            .zip(self.maximum.as_deref())
            .is_some_and(|(minimum, maximum)| minimum <= value && value <= maximum)
    }

    /// Returns whether the retained interval fully covers an ordered,
    /// inclusive query interval.
    #[must_use]
    pub fn covers_range(&self, lower: &[u8], upper: &[u8]) -> bool {
        lower <= upper
            && self
                .minimum
                .as_deref()
                .zip(self.maximum.as_deref())
                .is_some_and(|(minimum, maximum)| minimum <= lower && upper <= maximum)
    }

    /// Returns whether the retained interval overlaps an ordered, inclusive
    /// query interval.
    #[must_use]
    pub fn may_overlap_range(&self, lower: &[u8], upper: &[u8]) -> bool {
        lower <= upper
            && self
                .minimum
                .as_deref()
                .zip(self.maximum.as_deref())
                .is_some_and(|(minimum, maximum)| minimum <= upper && lower <= maximum)
    }

    fn validate_value(&self, value: &[u8]) -> Result<(), ZoneMapError> {
        if value.len() > self.profile.max_value_bytes {
            return Err(ZoneMapError::ValueLimitExceeded {
                actual: value.len(),
                limit: self.profile.max_value_bytes,
            });
        }
        Ok(())
    }

    fn checked_combined_count(&self, additional: u64) -> Result<u64, ZoneMapError> {
        let combined = self
            .count
            .checked_add(additional)
            .ok_or(ZoneMapError::CountOverflow)?;
        if combined > self.profile.max_observations {
            return Err(ZoneMapError::ObservationLimitExceeded {
                maximum: self.profile.max_observations,
            });
        }
        Ok(combined)
    }
}

fn try_copy_endpoint(value: &[u8], endpoint: ZoneMapAllocation) -> Result<Vec<u8>, ZoneMapError> {
    let mut owned = Vec::new();
    owned
        .try_reserve_exact(value.len())
        .map_err(|_: TryReserveError| ZoneMapError::AllocationFailed {
            endpoint,
            requested_bytes: value.len(),
        })?;
    owned.extend_from_slice(value);
    Ok(owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> ZoneMapProfile {
        ZoneMapProfile {
            max_value_bytes: 16,
            max_observations: 100,
        }
    }

    fn map(values: &[&[u8]]) -> ByteZoneMap {
        let mut map = ByteZoneMap::new(profile());
        for &value in values {
            map.try_observe(value).expect("bounded fixture");
        }
        map
    }

    #[test]
    fn byte_order_extrema_and_containment_are_canonical() {
        let mut zone = ByteZoneMap::new(profile());
        for value in [
            &b"middle"[..],
            &b""[..],
            &b"mid"[..],
            &b"\xff"[..],
            &b"z"[..],
        ] {
            zone.try_observe(value).expect("bounded fixture");
        }
        assert_eq!(zone.len(), 5);
        assert_eq!(zone.minimum(), Some(&b""[..]));
        assert_eq!(zone.maximum(), Some(&b"\xff"[..]));
        assert!(zone.may_contain(b"alpha"));
        assert!(zone.may_contain(b"\xff"));
        assert!(zone.covers_range(b"mid", b"z"));
        assert!(zone.may_overlap_range(b"z", b"\xff\xff"));
        assert!(!zone.covers_range(b"z", b"a"));
        assert!(!zone.may_overlap_range(b"\xff\x01", b"\xff\xff"));

        let empty = ByteZoneMap::new(profile());
        assert!(!empty.may_contain(b""));
        assert!(!empty.covers_range(b"", b""));
    }

    #[test]
    fn observation_limits_fail_without_mutation() {
        let mut zone = ByteZoneMap::new(ZoneMapProfile {
            max_value_bytes: 2,
            max_observations: 2,
        });
        zone.try_observe(b"b").expect("first value");
        let before = zone.clone();
        assert_eq!(
            zone.try_observe(b"long"),
            Err(ZoneMapError::ValueLimitExceeded {
                actual: 4,
                limit: 2,
            })
        );
        assert_eq!(zone, before);
        zone.try_observe(b"a").expect("second value");
        let before = zone.clone();
        assert_eq!(
            zone.try_observe(b"c"),
            Err(ZoneMapError::ObservationLimitExceeded { maximum: 2 })
        );
        assert_eq!(zone, before);
    }

    #[test]
    fn identical_profile_merge_is_order_independent() {
        let left = map(&[b"m", b"z"]);
        let right = map(&[b"a", b"x", b"y"]);
        let mut left_right = left.clone();
        left_right.try_merge(&right).expect("identical profile");
        let mut right_left = right;
        right_left.try_merge(&left).expect("identical profile");
        assert_eq!(left_right, right_left);
        assert_eq!(left_right.len(), 5);
        assert_eq!(left_right.minimum(), Some(&b"a"[..]));
        assert_eq!(left_right.maximum(), Some(&b"z"[..]));

        let mut identity = ByteZoneMap::new(profile());
        identity.try_merge(&left_right).expect("identical profile");
        assert_eq!(identity, left_right);
    }

    #[test]
    fn merge_profile_and_count_failures_are_atomic() {
        let mut zone = ByteZoneMap::new(ZoneMapProfile {
            max_value_bytes: 8,
            max_observations: 2,
        });
        zone.try_observe(b"m").expect("within limit");
        let different = ByteZoneMap::new(ZoneMapProfile {
            max_value_bytes: 9,
            max_observations: 2,
        });
        let before = zone.clone();
        assert_eq!(
            zone.try_merge(&different),
            Err(ZoneMapError::ProfileMismatch {
                left: ZoneMapProfile {
                    max_value_bytes: 8,
                    max_observations: 2,
                },
                right: ZoneMapProfile {
                    max_value_bytes: 9,
                    max_observations: 2,
                },
            })
        );
        assert_eq!(zone, before);

        let other = {
            let mut other = ByteZoneMap::new(zone.profile());
            other.try_observe(b"a").expect("within limit");
            other.try_observe(b"z").expect("within limit");
            other
        };
        assert_eq!(
            zone.try_merge(&other),
            Err(ZoneMapError::ObservationLimitExceeded { maximum: 2 })
        );
        assert_eq!(zone, before);
    }

    #[test]
    fn deletes_are_exact_when_extrema_are_derivable() {
        let mut repeated = map(&[b"k", b"k", b"k"]);
        repeated.try_remove(b"k").expect("same extrema remain");
        assert_eq!(repeated.len(), 2);
        assert_eq!(repeated.minimum(), Some(&b"k"[..]));
        repeated.try_remove(b"k").expect("same extrema remain");
        repeated.try_remove(b"k").expect("last value empties map");
        assert!(repeated.is_empty());
        assert_eq!(repeated.minimum(), None);
        assert_eq!(repeated.maximum(), None);

        let mut distinct = map(&[b"a", b"m", b"z"]);
        distinct.try_remove(b"m").expect("interior removal");
        assert_eq!(distinct.len(), 2);
        distinct.try_remove(b"a").expect("two endpoints are exact");
        assert_eq!(distinct.len(), 1);
        assert_eq!(distinct.minimum(), Some(&b"z"[..]));
        assert_eq!(distinct.maximum(), Some(&b"z"[..]));
    }

    #[test]
    fn ambiguous_endpoint_delete_requires_rebuild_atomically() {
        let mut zone = map(&[b"a", b"m", b"z", b"z"]);
        let before = zone.clone();
        assert_eq!(
            zone.try_remove(b"a"),
            Err(ZoneMapError::RebuildRequired {
                endpoint: ZoneMapEndpoint::Minimum,
                count: 4,
            })
        );
        assert_eq!(zone, before);
        assert_eq!(
            zone.try_remove(b"z"),
            Err(ZoneMapError::RebuildRequired {
                endpoint: ZoneMapEndpoint::Maximum,
                count: 4,
            })
        );
        assert_eq!(zone, before);
    }

    #[test]
    fn impossible_deletes_are_rejected_without_drift() {
        let mut empty = ByteZoneMap::new(profile());
        assert_eq!(
            empty.try_remove(b"a"),
            Err(ZoneMapError::MissingObservation)
        );

        let mut zone = map(&[b"a", b"z"]);
        let before = zone.clone();
        assert_eq!(zone.try_remove(b"m"), Err(ZoneMapError::MissingObservation));
        assert_eq!(
            zone.try_remove(b"\xff"),
            Err(ZoneMapError::MissingObservation)
        );
        assert_eq!(zone, before);
    }
}
