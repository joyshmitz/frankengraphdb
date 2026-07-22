//! Core value/type spine of FrankenGraphDB (bead `fgdb-w1-foundation-types-tjk`).
//!
//! Pure types only: no I/O, no durable state, `forbid(unsafe_code)`. Durable
//! *encoders* for these types are generated from the registries by
//! `fgdb-w1-generated-parsers` — the encodings here are the canonical value
//! encodings required for FG-INV-12 coherence (equality ⇒ same hash; total
//! order; encode/decode round trip), not wire formats.

#![forbid(unsafe_code)]

pub mod bytes;
pub mod ids;
pub mod refs;
pub mod scalar;

pub use bytes::{BoundedBytes, BoundedBytesError};
pub use ids::{
    BranchId, CommitSeq, DatabaseId, DatabaseSecurityNamespaceId, EId, GraphId, ObjectId,
    ServiceVisibilityEpoch, VId, WriterFenceEpoch,
};
pub use refs::{
    CommandRef, ConditionalCoordinateRef, ConditionalMarkerAxis, ConditionalMarkerRef,
    LogicalObjectKind, MarkerRef, StrongRef, WeakDigest,
};
pub use scalar::{CanonicalF64, CanonicalScalar, ScalarDecodeError};
