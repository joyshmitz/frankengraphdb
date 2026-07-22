//! Canonical scalar values under the `STRICT_PORTABLE` profile (plan §8.6).
//!
//! This module carries the FG-INV-12 binding clause: *canonical scalar
//! equality, hashing, ordering, and encoding are coherent*. Concretely:
//!
//! - `a == b` ⇒ `hash(a) == hash(b)` ⇒ `encode(a) == encode(b)`,
//! - `Ord` is total and transitive over all scalars (cross-type order is the
//!   fixed profile type rank: Null < Bool < Int < Float < Text < Bytes),
//! - `decode(encode(v)) == v` for every value, and decode rejects malformed
//!   input under length-before-allocation bounds.
//!
//! `STRICT_PORTABLE` float canonicalization: every NaN collapses to the one
//! canonical quiet NaN, `-0.0` collapses to `+0.0`, and the float order is
//! numeric with the canonical NaN greatest. Decimal / timestamp / pinned
//! collation scalars are later slices of the same bead
//! (`fgdb-w1-foundation-types-tjk`) — they extend this enum; they do not get
//! a second one.

use std::cmp::Ordering;

/// The single canonical quiet-NaN bit pattern under `STRICT_PORTABLE`.
const CANONICAL_NAN_BITS: u64 = 0x7FF8_0000_0000_0000;

/// Maximum text/bytes payload this profile accepts (length-before-allocation
/// guard; per-kind durable maxima can only be tighter).
pub const MAX_SCALAR_PAYLOAD: usize = 64 * 1024 * 1024;

/// An `f64` in canonical form: unique NaN, no negative zero. Constructing
/// one is the only way floats enter the scalar domain, so equality, hashing,
/// ordering, and encoding all see canonical bits only.
#[derive(Clone, Copy, Debug)]
pub struct CanonicalF64(u64);

impl CanonicalF64 {
    pub fn new(v: f64) -> Self {
        if v.is_nan() {
            return CanonicalF64(CANONICAL_NAN_BITS);
        }
        if v == 0.0 {
            // Collapses -0.0; 0.0f64.to_bits() is the +0 pattern.
            return CanonicalF64(0);
        }
        CanonicalF64(v.to_bits())
    }

    pub fn get(&self) -> f64 {
        f64::from_bits(self.0)
    }

    pub const fn to_bits(&self) -> u64 {
        self.0
    }

    /// Rejects non-canonical bit patterns instead of re-canonicalizing:
    /// durable inputs must already be canonical (fail closed, never repair).
    pub fn from_bits_canonical(bits: u64) -> Option<Self> {
        let v = f64::from_bits(bits);
        let canonical = if v.is_nan() {
            bits == CANONICAL_NAN_BITS
        } else {
            !(v == 0.0 && bits != 0)
        };
        canonical.then_some(CanonicalF64(bits))
    }
}

impl PartialEq for CanonicalF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for CanonicalF64 {}
impl std::hash::Hash for CanonicalF64 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}
impl PartialOrd for CanonicalF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for CanonicalF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        // total_cmp is IEEE totalOrder; with the single positive quiet NaN it
        // is numeric order with NaN greatest, and -0 cannot occur.
        self.get().total_cmp(&other.get())
    }
}

/// The canonical scalar union (current slice: the non-decimal, non-temporal
/// subset).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum CanonicalScalar {
    Null,
    Bool(bool),
    Int(i64),
    Float(CanonicalF64),
    Text(String),
    Bytes(Vec<u8>),
}

impl CanonicalScalar {
    /// Fixed cross-type rank of the `STRICT_PORTABLE` profile.
    fn type_rank(&self) -> u8 {
        match self {
            CanonicalScalar::Null => 0,
            CanonicalScalar::Bool(_) => 1,
            CanonicalScalar::Int(_) => 2,
            CanonicalScalar::Float(_) => 3,
            CanonicalScalar::Text(_) => 4,
            CanonicalScalar::Bytes(_) => 5,
        }
    }

    /// Canonical value encoding: `tag ‖ payload`. One encoding per value —
    /// uniqueness follows from fixed-width numerics and length-prefixed
    /// variable payloads over already-canonical in-memory forms.
    pub fn encode(&self) -> Vec<u8> {
        match self {
            CanonicalScalar::Null => vec![0x00],
            CanonicalScalar::Bool(b) => vec![0x01, u8::from(*b)],
            CanonicalScalar::Int(v) => {
                let mut out = vec![0x02];
                out.extend_from_slice(&v.to_le_bytes());
                out
            }
            CanonicalScalar::Float(fv) => {
                let mut out = vec![0x03];
                out.extend_from_slice(&fv.to_bits().to_le_bytes());
                out
            }
            CanonicalScalar::Text(s) => {
                let mut out = vec![0x04];
                out.extend_from_slice(&(s.len() as u64).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
                out
            }
            CanonicalScalar::Bytes(b) => {
                let mut out = vec![0x05];
                out.extend_from_slice(&(b.len() as u64).to_le_bytes());
                out.extend_from_slice(b);
                out
            }
        }
    }

    /// Decodes one scalar, consuming the entire input. Every malformed form
    /// is a typed rejection; declared lengths are validated against both the
    /// profile bound and the actually-present input before any allocation.
    pub fn decode(bytes: &[u8]) -> Result<Self, ScalarDecodeError> {
        let (&tag, rest) = bytes.split_first().ok_or(ScalarDecodeError::Empty)?;
        let exact = |want: usize| -> Result<&[u8], ScalarDecodeError> {
            if rest.len() != want {
                return Err(ScalarDecodeError::WrongPayloadLength {
                    tag,
                    expected: want,
                    got: rest.len(),
                });
            }
            Ok(rest)
        };
        let var = || -> Result<&[u8], ScalarDecodeError> {
            let (len_raw, body) = rest
                .split_first_chunk::<8>()
                .ok_or(ScalarDecodeError::TruncatedLength { tag })?;
            let declared = u64::from_le_bytes(*len_raw);
            let declared_usize = usize::try_from(declared)
                .map_err(|_| ScalarDecodeError::LengthOverflow { tag, declared })?;
            if declared_usize > MAX_SCALAR_PAYLOAD {
                return Err(ScalarDecodeError::LengthOverflow { tag, declared });
            }
            if body.len() != declared_usize {
                return Err(ScalarDecodeError::WrongPayloadLength {
                    tag,
                    expected: declared_usize,
                    got: body.len(),
                });
            }
            Ok(body)
        };
        match tag {
            0x00 => {
                exact(0)?;
                Ok(CanonicalScalar::Null)
            }
            0x01 => match exact(1)?[0] {
                0 => Ok(CanonicalScalar::Bool(false)),
                1 => Ok(CanonicalScalar::Bool(true)),
                other => Err(ScalarDecodeError::BadBool(other)),
            },
            0x02 => {
                let raw: [u8; 8] = exact(8)?.try_into().expect("length checked");
                Ok(CanonicalScalar::Int(i64::from_le_bytes(raw)))
            }
            0x03 => {
                let raw: [u8; 8] = exact(8)?.try_into().expect("length checked");
                let bits = u64::from_le_bytes(raw);
                CanonicalF64::from_bits_canonical(bits)
                    .map(CanonicalScalar::Float)
                    .ok_or(ScalarDecodeError::NonCanonicalFloat { bits })
            }
            0x04 => {
                let body = var()?;
                let s = std::str::from_utf8(body).map_err(|_| ScalarDecodeError::InvalidUtf8)?;
                Ok(CanonicalScalar::Text(s.to_owned()))
            }
            0x05 => Ok(CanonicalScalar::Bytes(var()?.to_vec())),
            other => Err(ScalarDecodeError::UnknownTag(other)),
        }
    }
}

impl PartialOrd for CanonicalScalar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CanonicalScalar {
    fn cmp(&self, other: &Self) -> Ordering {
        use CanonicalScalar as S;
        match (self, other) {
            (S::Null, S::Null) => Ordering::Equal,
            (S::Bool(a), S::Bool(b)) => a.cmp(b),
            (S::Int(a), S::Int(b)) => a.cmp(b),
            (S::Float(a), S::Float(b)) => a.cmp(b),
            (S::Text(a), S::Text(b)) => a.as_bytes().cmp(b.as_bytes()),
            (S::Bytes(a), S::Bytes(b)) => a.cmp(b),
            _ => self.type_rank().cmp(&other.type_rank()),
        }
    }
}

/// Typed rejections from [`CanonicalScalar::decode`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScalarDecodeError {
    Empty,
    UnknownTag(u8),
    BadBool(u8),
    TruncatedLength {
        tag: u8,
    },
    LengthOverflow {
        tag: u8,
        declared: u64,
    },
    WrongPayloadLength {
        tag: u8,
        expected: usize,
        got: usize,
    },
    NonCanonicalFloat {
        bits: u64,
    },
    InvalidUtf8,
}

impl std::fmt::Display for ScalarDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarDecodeError::Empty => write!(f, "empty scalar encoding"),
            ScalarDecodeError::UnknownTag(t) => write!(f, "unknown scalar tag {t:#04x}"),
            ScalarDecodeError::BadBool(b) => write!(f, "bool payload {b:#04x} is not 0/1"),
            ScalarDecodeError::TruncatedLength { tag } => {
                write!(f, "tag {tag:#04x}: truncated length prefix")
            }
            ScalarDecodeError::LengthOverflow { tag, declared } => {
                write!(
                    f,
                    "tag {tag:#04x}: declared length {declared} exceeds profile bound"
                )
            }
            ScalarDecodeError::WrongPayloadLength { tag, expected, got } => {
                write!(
                    f,
                    "tag {tag:#04x}: payload length {got}, expected {expected}"
                )
            }
            ScalarDecodeError::NonCanonicalFloat { bits } => {
                write!(
                    f,
                    "float bits {bits:#018x} are not STRICT_PORTABLE-canonical"
                )
            }
            ScalarDecodeError::InvalidUtf8 => write!(f, "text payload is not valid UTF-8"),
        }
    }
}

impl std::error::Error for ScalarDecodeError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn scalar(&mut self) -> CanonicalScalar {
            match self.next() % 6 {
                0 => CanonicalScalar::Null,
                1 => CanonicalScalar::Bool(self.next() % 2 == 1),
                2 => CanonicalScalar::Int(self.next() as i64),
                3 => CanonicalScalar::Float(CanonicalF64::new(f64::from_bits(self.next()))),
                4 => {
                    let n = (self.next() % 12) as usize;
                    CanonicalScalar::Text(
                        (0..n)
                            .map(|_| char::from(b'a' + (self.next() % 26) as u8))
                            .collect(),
                    )
                }
                _ => {
                    let n = (self.next() % 12) as usize;
                    CanonicalScalar::Bytes((0..n).map(|_| self.next() as u8).collect())
                }
            }
        }
    }

    fn hash_of(v: &CanonicalScalar) -> u64 {
        let mut h = DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }

    #[test]
    fn strict_portable_float_canonicalization() {
        // Every NaN collapses to the one canonical NaN.
        let nans = [
            f64::NAN,
            -f64::NAN,
            f64::from_bits(0x7FF0_0000_0000_0001),
            f64::from_bits(0xFFF8_0000_0000_1234),
        ];
        for n in nans {
            assert_eq!(
                CanonicalF64::new(n).to_bits(),
                CANONICAL_NAN_BITS,
                "{:#x}",
                n.to_bits()
            );
        }
        // -0.0 collapses to +0.0 and compares equal.
        assert_eq!(CanonicalF64::new(-0.0), CanonicalF64::new(0.0));
        assert_eq!(CanonicalF64::new(-0.0).to_bits(), 0);
        // NaN sorts greatest; the rest is numeric order.
        let mut vals: Vec<CanonicalF64> =
            [f64::NAN, 1.0, f64::NEG_INFINITY, -0.0, f64::INFINITY, -1.5]
                .into_iter()
                .map(CanonicalF64::new)
                .collect();
        vals.sort();
        let got: Vec<f64> = vals.iter().map(CanonicalF64::get).collect();
        assert_eq!(got[0], f64::NEG_INFINITY);
        assert_eq!(got[1], -1.5);
        assert_eq!(got[2], 0.0);
        assert_eq!(got[3], 1.0);
        assert_eq!(got[4], f64::INFINITY);
        assert!(got[5].is_nan());
    }

    #[test]
    fn noncanonical_float_bits_are_rejected_not_repaired() {
        assert!(CanonicalF64::from_bits_canonical(CANONICAL_NAN_BITS).is_some());
        assert!(CanonicalF64::from_bits_canonical(0x7FF0_0000_0000_0001).is_none());
        assert!(CanonicalF64::from_bits_canonical((-0.0f64).to_bits()).is_none());
        assert!(CanonicalF64::from_bits_canonical(1.5f64.to_bits()).is_some());
    }

    #[test]
    fn equal_implies_same_hash_and_same_encoding() {
        for seed in [1u64, 0xFEED, 0x00C0FFEE] {
            let mut rng = SplitMix64(seed);
            for _ in 0..500 {
                let a = rng.scalar();
                let b = a.clone();
                assert_eq!(a, b);
                assert_eq!(hash_of(&a), hash_of(&b), "seed={seed} value={a:?}");
                assert_eq!(a.encode(), b.encode(), "seed={seed} value={a:?}");
            }
        }
    }

    #[test]
    fn encode_decode_round_trips_all_variants() {
        for seed in [2u64, 0xDECAF, u64::MAX / 3] {
            let mut rng = SplitMix64(seed);
            for _ in 0..500 {
                let v = rng.scalar();
                let enc = v.encode();
                let back = CanonicalScalar::decode(&enc).unwrap_or_else(|e| {
                    panic!("seed={seed} decode({enc:02x?}) of {v:?} failed: {e}")
                });
                assert_eq!(back, v, "seed={seed} enc={enc:02x?}");
            }
        }
    }

    #[test]
    fn ordering_is_total_transitive_and_type_ranked() {
        for seed in [5u64, 0xBEEF] {
            let mut rng = SplitMix64(seed);
            for _ in 0..400 {
                let (a, b, c) = (rng.scalar(), rng.scalar(), rng.scalar());
                // Totality: cmp never panics and is antisymmetric.
                assert_eq!(a.cmp(&b).reverse(), b.cmp(&a), "seed={seed} {a:?} {b:?}");
                // Transitivity via sort correctness on the triple.
                let mut v = [a.clone(), b.clone(), c.clone()];
                v.sort();
                assert!(
                    v[0] <= v[1] && v[1] <= v[2] && v[0] <= v[2],
                    "seed={seed} {v:?}"
                );
            }
        }
        assert!(CanonicalScalar::Null < CanonicalScalar::Bool(false));
        assert!(CanonicalScalar::Bool(true) < CanonicalScalar::Int(i64::MIN));
        assert!(CanonicalScalar::Int(i64::MAX) < CanonicalScalar::Float(CanonicalF64::new(0.0)));
        assert!(
            CanonicalScalar::Float(CanonicalF64::new(f64::INFINITY))
                < CanonicalScalar::Text(String::new())
        );
        assert!(CanonicalScalar::Text("zzz".into()) < CanonicalScalar::Bytes(vec![]));
    }

    #[test]
    fn malformed_decodes_are_typed_rejections() {
        use ScalarDecodeError as E;
        assert_eq!(CanonicalScalar::decode(&[]), Err(E::Empty));
        assert_eq!(CanonicalScalar::decode(&[0x77]), Err(E::UnknownTag(0x77)));
        assert_eq!(CanonicalScalar::decode(&[0x01, 2]), Err(E::BadBool(2)));
        assert_eq!(
            CanonicalScalar::decode(&[0x00, 0xAA]),
            Err(E::WrongPayloadLength {
                tag: 0x00,
                expected: 0,
                got: 1
            })
        );
        assert_eq!(
            CanonicalScalar::decode(&[0x04, 1, 0, 0]),
            Err(E::TruncatedLength { tag: 0x04 })
        );
        // Declared length far past the profile bound: rejected before any
        // allocation even though the body is absent.
        let mut huge = vec![0x05];
        huge.extend_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(
            CanonicalScalar::decode(&huge),
            Err(E::LengthOverflow {
                tag: 0x05,
                declared: u64::MAX
            })
        );
        // Declared length not matching the actual body.
        let mut short = vec![0x05];
        short.extend_from_slice(&4u64.to_le_bytes());
        short.extend_from_slice(&[1, 2]);
        assert_eq!(
            CanonicalScalar::decode(&short),
            Err(E::WrongPayloadLength {
                tag: 0x05,
                expected: 4,
                got: 2
            })
        );
        // Non-canonical float bits in a durable image: fail closed.
        let mut nc = vec![0x03];
        nc.extend_from_slice(&0x7FF0_0000_0000_0001u64.to_le_bytes());
        assert_eq!(
            CanonicalScalar::decode(&nc),
            Err(E::NonCanonicalFloat {
                bits: 0x7FF0_0000_0000_0001
            })
        );
        // Invalid UTF-8 text.
        let mut bad = vec![0x04];
        bad.extend_from_slice(&2u64.to_le_bytes());
        bad.extend_from_slice(&[0xFF, 0xFE]);
        assert_eq!(CanonicalScalar::decode(&bad), Err(E::InvalidUtf8));
    }
}
