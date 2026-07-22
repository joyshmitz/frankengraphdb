//! Length-bounded byte containers.
//!
//! Appendix A's encoding preamble requires "length-before-allocation" and
//! per-kind maximum sizes. `BoundedBytes<MAX>` is the type-level form of that
//! rule: the bound is part of the type, construction checks it, and no
//! constructor can allocate past it.

/// Owned bytes whose length is statically bounded by `MAX`.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct BoundedBytes<const MAX: usize> {
    data: Vec<u8>,
}

/// Rejection produced when a byte string exceeds its declared bound.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BoundedBytesError {
    pub declared_len: usize,
    pub max: usize,
}

impl std::fmt::Display for BoundedBytesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "byte string of length {} exceeds declared bound {}",
            self.declared_len, self.max
        )
    }
}

impl std::error::Error for BoundedBytesError {}

impl<const MAX: usize> BoundedBytes<MAX> {
    /// Takes ownership of `data` if it fits the bound.
    pub fn new(data: Vec<u8>) -> Result<Self, BoundedBytesError> {
        if data.len() > MAX {
            return Err(BoundedBytesError {
                declared_len: data.len(),
                max: MAX,
            });
        }
        Ok(BoundedBytes { data })
    }

    /// Length-before-allocation construction: validates `declared_len`
    /// against both the bound and the actually-available input *before*
    /// copying, then copies exactly `declared_len` bytes.
    pub fn from_declared_len(declared_len: usize, input: &[u8]) -> Result<Self, BoundedBytesError> {
        if declared_len > MAX || declared_len > input.len() {
            return Err(BoundedBytesError {
                declared_len,
                max: MAX,
            });
        }
        Ok(BoundedBytes {
            data: input[..declared_len].to_vec(),
        })
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub const fn max_len() -> usize {
        MAX
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bound_is_enforced_at_construction() {
        assert!(BoundedBytes::<4>::new(vec![1, 2, 3, 4]).is_ok());
        let err = BoundedBytes::<4>::new(vec![0; 5]).unwrap_err();
        assert_eq!(
            err,
            BoundedBytesError {
                declared_len: 5,
                max: 4
            }
        );
    }

    #[test]
    fn declared_len_is_checked_before_any_copy() {
        // Declared length past the bound: rejected even though input is short.
        assert!(BoundedBytes::<4>::from_declared_len(usize::MAX, &[1, 2]).is_err());
        // Declared length past the available input: rejected (no partial read).
        assert!(BoundedBytes::<8>::from_declared_len(3, &[1, 2]).is_err());
        // Exact prefix taken otherwise.
        let ok = BoundedBytes::<8>::from_declared_len(2, &[1, 2, 3]).unwrap();
        assert_eq!(ok.as_slice(), &[1, 2]);
    }
}
