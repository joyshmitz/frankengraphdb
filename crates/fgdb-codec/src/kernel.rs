//! Honest dispatch seams for the safe codec kernels.
//!
//! This module currently exposes exactly one implementation path:
//! [`DispatchPath::Scalar`]. It does not probe the host or imply that a SIMD
//! implementation exists. Future dispatch work can add a separately verified
//! path without changing callers from direct function calls a second time.
//!
//! These traits select reusable kernel mechanics only. They do not select a
//! durable codec identifier, format envelope, or adaptive policy.

#![forbid(unsafe_code)]

use crate::{bitpack, block};

/// Implementation path used by one codec operation.
///
/// The closed enum is intentionally honest about the implementation currently
/// available in this crate.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DispatchPath {
    /// Portable, safe scalar Rust.
    Scalar,
}

impl DispatchPath {
    /// Returns the stable symbolic label used in diagnostic evidence.
    ///
    /// This label is not a durable codec identifier.
    #[must_use]
    pub const fn evidence_label(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
        }
    }
}

/// Checked fixed-width and frame-of-reference bitpacking operations.
pub trait BitpackKernel {
    /// Implementation path used by this kernel.
    const DISPATCH_PATH: DispatchPath;

    /// Encodes fixed-width values into canonical bytes.
    fn encode(&self, values: &[u64], width: u8) -> Result<Vec<u8>, bitpack::BitpackError>;

    /// Encodes fixed-width values into caller-owned storage.
    fn encode_into(
        &self,
        values: &[u64],
        width: u8,
        output: &mut [u8],
    ) -> Result<usize, bitpack::BitpackError>;

    /// Decodes exactly `count` fixed-width values.
    fn decode(
        &self,
        input: &[u8],
        count: usize,
        width: u8,
    ) -> Result<Vec<u64>, bitpack::BitpackError>;

    /// Frame-of-reference encodes values relative to `base`.
    fn encode_for(
        &self,
        values: &[u64],
        base: u64,
        width: u8,
    ) -> Result<Vec<u8>, bitpack::BitpackError>;

    /// Decodes frame-of-reference values relative to `base`.
    fn decode_for(
        &self,
        input: &[u8],
        count: usize,
        base: u64,
        width: u8,
    ) -> Result<Vec<u64>, bitpack::BitpackError>;
}

/// Checked deterministic block compression and decompression operations.
pub trait BlockKernel {
    /// Implementation path used by this kernel.
    const DISPATCH_PATH: DispatchPath;

    /// Compresses one caller-framed block under an immutable scalar profile.
    fn compress(
        &self,
        input: &[u8],
        profile: block::CodecProfile,
    ) -> Result<Vec<u8>, block::CompressionError>;

    /// Decompresses one token stream to an exact authenticated length.
    fn decompress(
        &self,
        input: &[u8],
        expected_decoded_len: usize,
        output_limit: block::OutputLimit,
    ) -> Result<Vec<u8>, block::DecodeError>;
}

/// Zero-sized selector for the crate's safe scalar kernels.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ScalarKernels;

impl ScalarKernels {
    /// Returns the only implementation path currently available.
    #[must_use]
    pub const fn dispatch_path(self) -> DispatchPath {
        DispatchPath::Scalar
    }
}

impl BitpackKernel for ScalarKernels {
    const DISPATCH_PATH: DispatchPath = DispatchPath::Scalar;

    fn encode(&self, values: &[u64], width: u8) -> Result<Vec<u8>, bitpack::BitpackError> {
        bitpack::encode(values, width)
    }

    fn encode_into(
        &self,
        values: &[u64],
        width: u8,
        output: &mut [u8],
    ) -> Result<usize, bitpack::BitpackError> {
        bitpack::encode_into(values, width, output)
    }

    fn decode(
        &self,
        input: &[u8],
        count: usize,
        width: u8,
    ) -> Result<Vec<u64>, bitpack::BitpackError> {
        bitpack::decode(input, count, width)
    }

    fn encode_for(
        &self,
        values: &[u64],
        base: u64,
        width: u8,
    ) -> Result<Vec<u8>, bitpack::BitpackError> {
        bitpack::encode_for(values, base, width)
    }

    fn decode_for(
        &self,
        input: &[u8],
        count: usize,
        base: u64,
        width: u8,
    ) -> Result<Vec<u64>, bitpack::BitpackError> {
        bitpack::decode_for(input, count, base, width)
    }
}

impl BlockKernel for ScalarKernels {
    const DISPATCH_PATH: DispatchPath = DispatchPath::Scalar;

    fn compress(
        &self,
        input: &[u8],
        profile: block::CodecProfile,
    ) -> Result<Vec<u8>, block::CompressionError> {
        block::compress(input, profile)
    }

    fn decompress(
        &self,
        input: &[u8],
        expected_decoded_len: usize,
        output_limit: block::OutputLimit,
    ) -> Result<Vec<u8>, block::DecodeError> {
        block::decompress(input, expected_decoded_len, output_limit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KERNELS: ScalarKernels = ScalarKernels;

    #[test]
    fn selector_is_zero_sized_and_reports_only_scalar() {
        assert_eq!(core::mem::size_of::<ScalarKernels>(), 0);
        assert_eq!(KERNELS.dispatch_path(), DispatchPath::Scalar);
        assert_eq!(
            <ScalarKernels as BitpackKernel>::DISPATCH_PATH,
            DispatchPath::Scalar
        );
        assert_eq!(
            <ScalarKernels as BlockKernel>::DISPATCH_PATH,
            DispatchPath::Scalar
        );
        assert_eq!(DispatchPath::Scalar.evidence_label(), "scalar");
    }

    #[test]
    fn bitpack_trait_matches_direct_bytes_decode_and_errors() {
        let values = [3_u64, 0, 17, 31, 9];
        let direct = bitpack::encode(&values, 5);
        let dispatched = BitpackKernel::encode(&KERNELS, &values, 5);
        assert_eq!(dispatched, direct);

        let encoded = direct.expect("valid fixed-width fixture");
        assert_eq!(
            BitpackKernel::decode(&KERNELS, &encoded, values.len(), 5),
            bitpack::decode(&encoded, values.len(), 5)
        );

        let mut direct_output = [0xa5_u8; 8];
        let mut dispatched_output = direct_output;
        assert_eq!(
            BitpackKernel::encode_into(&KERNELS, &values, 5, &mut dispatched_output),
            bitpack::encode_into(&values, 5, &mut direct_output)
        );
        assert_eq!(dispatched_output, direct_output);

        let invalid_values = [0_u64, 8];
        assert_eq!(
            BitpackKernel::encode(&KERNELS, &invalid_values, 3),
            bitpack::encode(&invalid_values, 3)
        );
        assert_eq!(
            BitpackKernel::decode(&KERNELS, &[0x80], 1, 1),
            bitpack::decode(&[0x80], 1, 1)
        );
    }

    #[test]
    fn bitpack_trait_matches_direct_for_operations() {
        let values = [1_000_u64, 1_003, 1_007, 1_015];
        let direct = bitpack::encode_for(&values, 1_000, 4);
        let dispatched = BitpackKernel::encode_for(&KERNELS, &values, 1_000, 4);
        assert_eq!(dispatched, direct);

        let encoded = direct.expect("valid frame-of-reference fixture");
        assert_eq!(
            BitpackKernel::decode_for(&KERNELS, &encoded, values.len(), 1_000, 4),
            bitpack::decode_for(&encoded, values.len(), 1_000, 4)
        );

        let invalid_values = [999_u64];
        assert_eq!(
            BitpackKernel::encode_for(&KERNELS, &invalid_values, 1_000, 4),
            bitpack::encode_for(&invalid_values, 1_000, 4)
        );
    }

    #[test]
    fn block_trait_matches_direct_bytes_decode_and_errors() {
        let profile =
            block::CodecProfile::try_new(4_096, 256, 4_096).expect("valid scalar profile");
        let input = b"abcdefghabcdefghabcdefgh:scalar-block-fixture";

        let direct = block::compress(input, profile);
        let dispatched = BlockKernel::compress(&KERNELS, input, profile);
        assert_eq!(dispatched, direct);

        let encoded = direct.expect("valid block fixture");
        let limit = block::OutputLimit::new(input.len());
        assert_eq!(
            BlockKernel::decompress(&KERNELS, &encoded, input.len(), limit),
            block::decompress(&encoded, input.len(), limit)
        );

        let too_large = vec![0_u8; profile.max_block_len() + 1];
        assert_eq!(
            BlockKernel::compress(&KERNELS, &too_large, profile),
            block::compress(&too_large, profile)
        );

        let malformed = [0x80_u8, 0, 0];
        assert_eq!(
            BlockKernel::decompress(&KERNELS, &malformed, 4, block::OutputLimit::new(4)),
            block::decompress(&malformed, 4, block::OutputLimit::new(4))
        );
    }
}
