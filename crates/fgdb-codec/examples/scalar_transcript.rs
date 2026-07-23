use std::{error::Error, fmt::Write as _};

use fgdb_codec::{
    block::{CodecProfile, OutputLimit},
    delta_varint::EntryLimit as DeltaVarintEntryLimit,
    elias_fano::EntryLimit,
    evidence::CodecRunRow,
    kernel::{
        BitpackKernel, BlockKernel, DeltaVarintKernel, EliasFanoKernel, KernelOutput,
        NeighborKernel, RoaringKernel, ScalarKernels, VarintKernel,
    },
    neighbor::NeighborCodec,
    roaring::EntryLimit as RoaringEntryLimit,
};

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn unexpected_success(message: &'static str) -> Box<dyn Error> {
    std::io::Error::other(message).into()
}

fn emit_evidence(
    codec_id: &str,
    corpus_id: &str,
    entry_count: usize,
    output: &KernelOutput,
) -> Result<(), Box<dyn Error>> {
    let row = CodecRunRow::try_from_kernel_output(codec_id, corpus_id, entry_count, output)?;
    print!("{}", row.to_ndjson()?);
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let kernels = ScalarKernels;
    println!("== fgdb-codec scalar transcript v1 ==");

    let max_varint = VarintKernel::encode_varint_output(&kernels, u64::MAX);
    emit_evidence("uleb128-scalar-diagnostic", "u64-max", 1, &max_varint)?;
    println!("uleb128 max: {}", hex(max_varint.as_bytes()));
    let nonminimal = match VarintKernel::decode_varint(&kernels, &[0x80, 0x00]) {
        Err(error) => error,
        Ok(_) => return Err(unexpected_success("nonminimal varint was accepted")),
    };
    println!("uleb128 reject nonminimal: {nonminimal}");

    let delta_values = [127, 127, 255, 16_384];
    let delta_encoded = DeltaVarintKernel::encode_delta_varint_output(&kernels, &delta_values)?;
    emit_evidence(
        "delta-varint-scalar-diagnostic",
        "scalar-delta-fixture",
        delta_values.len(),
        &delta_encoded,
    )?;
    println!(
        "delta_varint count=4: bytes={} decoded={:?}",
        hex(delta_encoded.as_bytes()),
        DeltaVarintKernel::decode_delta_varint(
            &kernels,
            delta_encoded.as_bytes(),
            delta_values.len(),
            DeltaVarintEntryLimit::new(delta_values.len()),
        )?
    );
    let decreasing_delta = match DeltaVarintKernel::encode_delta_varint_output(&kernels, &[8, 5]) {
        Err(error) => error,
        Ok(_) => {
            return Err(unexpected_success(
                "decreasing delta-varint input was accepted",
            ));
        }
    };
    println!("delta_varint reject decreasing: {decreasing_delta}");

    let block_profile = CodecProfile::try_new(4096, 256, 4096)?;
    let block_input = b"abcdabcdabcd";
    let block_encoded = BlockKernel::compress_output(&kernels, block_input, block_profile)?;
    emit_evidence(
        "block-scalar-diagnostic",
        "scalar-repetition-fixture",
        block_input.len(),
        &block_encoded,
    )?;
    println!(
        "block input={} encoded={} bytes={} decoded={}",
        block_input.len(),
        block_encoded.len(),
        hex(block_encoded.as_bytes()),
        String::from_utf8(BlockKernel::decompress(
            &kernels,
            block_encoded.as_bytes(),
            block_input.len(),
            OutputLimit::new(block_input.len()),
        )?)?
    );

    let packed_values = [0, 1, 2, 3, 4, 5, 30, 31];
    let packed = BitpackKernel::encode_output(&kernels, &packed_values, 5)?;
    emit_evidence(
        "bitpack-scalar-diagnostic",
        "scalar-width5-fixture",
        packed_values.len(),
        &packed,
    )?;
    println!(
        "bitpack width=5 count=8: bytes={} decoded={:?}",
        hex(packed.as_bytes()),
        BitpackKernel::decode(&kernels, packed.as_bytes(), packed_values.len(), 5)?
    );
    let nonzero_padding = match BitpackKernel::decode(&kernels, &[0x20], 1, 5) {
        Err(error) => error,
        Ok(_) => return Err(unexpected_success("nonzero bitpack padding was accepted")),
    };
    println!("bitpack reject nonzero padding: {nonzero_padding}");

    let frame_values = [100, 101, 105, 109, 115];
    let frame = BitpackKernel::encode_for_output(&kernels, &frame_values, 100, 4)?;
    emit_evidence(
        "for-bitpack-scalar-diagnostic",
        "scalar-base100-width4-fixture",
        frame_values.len(),
        &frame,
    )?;
    println!(
        "for base=100 width=4 count=5: bytes={} decoded={:?}",
        hex(frame.as_bytes()),
        BitpackKernel::decode_for(&kernels, frame.as_bytes(), frame_values.len(), 100, 4)?
    );

    let monotone = [0, 1, 1, 3, 5, 8, 13, 21, 34, 55];
    let ef =
        EliasFanoKernel::build_elias_fano(&kernels, &monotone, EntryLimit::new(monotone.len()))?;
    println!(
        "elias_fano count={} low_bits={} high_bits={} logical_storage_words={}",
        ef.len(),
        ef.low_bits(),
        ef.high_bit_len(),
        ef.logical_storage_words()
    );
    let selected = EliasFanoKernel::elias_fano_select(&kernels, &ef, 7)
        .ok_or_else(|| unexpected_success("Elias-Fano select lost an in-range value"))?;
    println!(
        "elias_fano rank_le(13)={} select(7)={selected}",
        EliasFanoKernel::elias_fano_rank_le(&kernels, &ef, 13)
    );
    println!(
        "elias_fano predecessor(20)={:?} successor(20)={:?}",
        EliasFanoKernel::elias_fano_predecessor(&kernels, &ef, 20),
        EliasFanoKernel::elias_fano_successor(&kernels, &ef, 20)
    );
    let decreasing =
        match EliasFanoKernel::build_elias_fano(&kernels, &[1, 4, 3], EntryLimit::new(3)) {
            Err(error) => error,
            Ok(_) => {
                return Err(unexpected_success(
                    "decreasing Elias-Fano input was accepted",
                ));
            }
        };
    println!("elias_fano reject decreasing: {decreasing}");

    let bitmap_values = [1, 2, 3, 10, 65_536, u32::MAX];
    let bitmap = RoaringKernel::build_roaring(
        &kernels,
        &bitmap_values,
        RoaringEntryLimit::new(bitmap_values.len()),
    )?;
    let bitmap_other =
        RoaringKernel::build_roaring(&kernels, &[2, 10, 65_536], RoaringEntryLimit::new(3))?;
    let bitmap_intersection = RoaringKernel::roaring_intersection(
        &kernels,
        &bitmap,
        &bitmap_other,
        RoaringEntryLimit::new(3),
    )?;
    println!(
        "roaring count={} chunks={} rank_le(10)={} select(4)={:?} intersection={:?}",
        bitmap.len(),
        bitmap.chunk_count(),
        RoaringKernel::roaring_rank_le(&kernels, &bitmap, 10),
        RoaringKernel::roaring_select(&kernels, &bitmap, 4),
        bitmap_intersection.iter().collect::<Vec<_>>()
    );

    let neighbors = [1, 2, 3, 10, 127, 128, 1_000];
    let stream = NeighborKernel::build_neighbors(
        &kernels,
        NeighborCodec::StreamVByte,
        &neighbors,
        EntryLimit::new(neighbors.len()),
    )?;
    let dense = NeighborKernel::build_neighbors(
        &kernels,
        NeighborCodec::DenseIntervals,
        &[2, 3, 4, 10, 11, 1_000],
        EntryLimit::new(6),
    )?;
    println!(
        "neighbor codec={:?} count={} rank_le(128)={} select(5)={:?} intersection={:?}",
        stream.codec(),
        stream.len(),
        NeighborKernel::neighbors_rank_le(&kernels, &stream, 128),
        NeighborKernel::neighbors_select(&kernels, &stream, 5),
        NeighborKernel::neighbors_intersection(&kernels, &stream, &dense, EntryLimit::new(4),)?
    );

    Ok(())
}
