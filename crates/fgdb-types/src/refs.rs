//! The typed reference family (Appendix A "Reference semantics", plan ~§1394).
//!
//! Every `ObjectId`-bearing edge in the durable graph declares its retention
//! semantics *in its type*:
//!
//! - [`StrongRef<T>`] — always followed; retains its target.
//! - [`ConditionalCoordinateRef<T>`] — sequence-neutral payload whose
//!   retention sequence comes from its enclosing committed marker; pending /
//!   prepared traversal treats it as strong because no committed-marker cut
//!   context exists yet.
//! - [`ConditionalMarkerRef`] — followed until an authenticated matching
//!   checkpoint/cut on its axis.
//! - [`WeakDigest<T>`] — comparison only; never a reachability edge.
//! - [`MarkerRef`] / [`CommandRef`] — **identities, not reachability by
//!   themselves** (the a01 law): they deliberately do *not* carry a type
//!   parameter or implement any traversal trait; an enclosing tagged
//!   reference supplies reachability.
//!
//! Type parameters are anchored to durable object kinds through
//! [`LogicalObjectKind`], whose `OBJECT_KIND` codes must match rows in
//! `registries/logical_object_kinds.toml`. Target types live in the crates
//! that own their formats; this crate only defines the reference machinery.

use std::marker::PhantomData;

use crate::ids::{BranchId, CommitSeq, GraphId, ObjectId};

/// Implemented by every durable logical object type that references can
/// target. `OBJECT_KIND` must equal the type's registered code in
/// `registries/logical_object_kinds.toml`; the registry-check tooling owns
/// cross-checking codes against the registry, and unit tests here only pin
/// local distinctness of whatever kinds are linked into a build.
pub trait LogicalObjectKind {
    const OBJECT_KIND: u16;
    const KIND_NAME: &'static str;
}

macro_rules! fmt_ref_debug {
    ($name:literal) => {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, concat!($name, "<{}>"), T::KIND_NAME)
        }
    };
}

/// Always-followed retaining reference to a `T`.
pub struct StrongRef<T: LogicalObjectKind> {
    oid: ObjectId,
    _target: PhantomData<fn() -> T>,
}

impl<T: LogicalObjectKind> StrongRef<T> {
    pub const fn new(oid: ObjectId) -> Self {
        StrongRef {
            oid,
            _target: PhantomData,
        }
    }

    pub const fn oid(&self) -> ObjectId {
        self.oid
    }

    /// The registered kind code of the target type.
    pub const fn target_kind() -> u16 {
        T::OBJECT_KIND
    }
}

// Manual impls: derive would bound them on `T: Clone` etc., but the phantom
// target type never affects the value.
impl<T: LogicalObjectKind> Clone for StrongRef<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: LogicalObjectKind> Copy for StrongRef<T> {}
impl<T: LogicalObjectKind> PartialEq for StrongRef<T> {
    fn eq(&self, other: &Self) -> bool {
        self.oid == other.oid
    }
}
impl<T: LogicalObjectKind> Eq for StrongRef<T> {}
impl<T: LogicalObjectKind> std::hash::Hash for StrongRef<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.oid.hash(state);
    }
}
impl<T: LogicalObjectKind> std::fmt::Debug for StrongRef<T> {
    fmt_ref_debug!("StrongRef");
}

/// Sequence-neutral conditional reference: the payload names its target and
/// branch coordinates, and the *enclosing committed marker* supplies the
/// retention sequence.
pub struct ConditionalCoordinateRef<T: LogicalObjectKind> {
    oid: ObjectId,
    graph: GraphId,
    branch: BranchId,
    _target: PhantomData<fn() -> T>,
}

impl<T: LogicalObjectKind> ConditionalCoordinateRef<T> {
    pub const fn new(oid: ObjectId, graph: GraphId, branch: BranchId) -> Self {
        ConditionalCoordinateRef {
            oid,
            graph,
            branch,
            _target: PhantomData,
        }
    }
    pub const fn oid(&self) -> ObjectId {
        self.oid
    }
    pub const fn graph(&self) -> GraphId {
        self.graph
    }
    pub const fn branch(&self) -> BranchId {
        self.branch
    }
}

impl<T: LogicalObjectKind> Clone for ConditionalCoordinateRef<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: LogicalObjectKind> Copy for ConditionalCoordinateRef<T> {}
impl<T: LogicalObjectKind> PartialEq for ConditionalCoordinateRef<T> {
    fn eq(&self, other: &Self) -> bool {
        (self.oid, self.graph, self.branch) == (other.oid, other.graph, other.branch)
    }
}
impl<T: LogicalObjectKind> Eq for ConditionalCoordinateRef<T> {}
impl<T: LogicalObjectKind> std::hash::Hash for ConditionalCoordinateRef<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.oid, self.graph, self.branch).hash(state);
    }
}
impl<T: LogicalObjectKind> std::fmt::Debug for ConditionalCoordinateRef<T> {
    fmt_ref_debug!("ConditionalCoordinateRef");
}

/// Comparison-only digest of a `T`; never followed, never retains.
pub struct WeakDigest<T: LogicalObjectKind> {
    digest: [u8; 32],
    _target: PhantomData<fn() -> T>,
}

impl<T: LogicalObjectKind> WeakDigest<T> {
    pub const fn new(digest: [u8; 32]) -> Self {
        WeakDigest {
            digest,
            _target: PhantomData,
        }
    }
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }
}

impl<T: LogicalObjectKind> Clone for WeakDigest<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: LogicalObjectKind> Copy for WeakDigest<T> {}
impl<T: LogicalObjectKind> PartialEq for WeakDigest<T> {
    fn eq(&self, other: &Self) -> bool {
        self.digest == other.digest
    }
}
impl<T: LogicalObjectKind> Eq for WeakDigest<T> {}
impl<T: LogicalObjectKind> std::hash::Hash for WeakDigest<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.digest.hash(state);
    }
}
impl<T: LogicalObjectKind> std::fmt::Debug for WeakDigest<T> {
    fmt_ref_debug!("WeakDigest");
}

/// Bare marker identity: `{marker_oid, commit_seq}`. **Not reachability by
/// itself** — an enclosing tagged reference supplies that (a01 law). This
/// type therefore exposes no `oid()`-style traversal accessor naming and no
/// target type parameter.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MarkerRef {
    pub marker_oid: ObjectId,
    pub commit_seq: CommitSeq,
}

/// Bare command identity: `{command_record_oid, logical_command_seq}`.
/// Like [`MarkerRef`], an identity — never a retention edge on its own.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CommandRef {
    pub command_record_oid: ObjectId,
    pub logical_command_seq: u64,
}

/// Cut axis for a [`ConditionalMarkerRef`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ConditionalMarkerAxis {
    /// Followed until a verified matching global checkpoint cut.
    Global,
    /// Followed until a verified matching cut on one branch's axis.
    Branch { graph: GraphId, branch: BranchId },
}

/// Marker reference followed until an authenticated matching checkpoint/cut
/// on its declared axis.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ConditionalMarkerRef {
    pub marker: MarkerRef,
    pub axis: ConditionalMarkerAxis,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    struct CommitCapsule;
    impl LogicalObjectKind for CommitCapsule {
        const OBJECT_KIND: u16 = 0x0001;
        const KIND_NAME: &'static str = "CommitCapsule";
    }

    struct CommitMarker;
    impl LogicalObjectKind for CommitMarker {
        const OBJECT_KIND: u16 = 0x0006;
        const KIND_NAME: &'static str = "CommitMarker";
    }

    fn oid(fill: u8) -> ObjectId {
        ObjectId([fill; 32])
    }

    #[test]
    fn strong_refs_to_different_kinds_are_different_types() {
        let a: StrongRef<CommitCapsule> = StrongRef::new(oid(1));
        let b: StrongRef<CommitMarker> = StrongRef::new(oid(1));
        // Same oid, different target kind: comparing them is a compile error
        // (uncomment to verify), and the kinds are observably distinct.
        // assert_eq!(a, b);
        assert_ne!(
            StrongRef::<CommitCapsule>::target_kind(),
            StrongRef::<CommitMarker>::target_kind()
        );
        assert_eq!(a.oid(), b.oid());
        assert_eq!(format!("{a:?}"), "StrongRef<CommitCapsule>");
    }

    #[test]
    fn reference_equality_and_hash_are_value_based() {
        let a: StrongRef<CommitCapsule> = StrongRef::new(oid(9));
        let b: StrongRef<CommitCapsule> = StrongRef::new(oid(9));
        assert_eq!(a, b);
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        a.hash(&mut h1);
        b.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn marker_and_command_identities_carry_no_traversal_surface() {
        // The a01 law in type form: MarkerRef/CommandRef expose only their
        // identity fields; only the enclosing tagged types add axis/traversal
        // meaning.
        let m = MarkerRef {
            marker_oid: oid(3),
            commit_seq: CommitSeq(41999),
        };
        let c = ConditionalMarkerRef {
            marker: m,
            axis: ConditionalMarkerAxis::Global,
        };
        assert_eq!(c.marker, m);
        let br = ConditionalMarkerRef {
            marker: m,
            axis: ConditionalMarkerAxis::Branch {
                graph: GraphId(1),
                branch: BranchId(2),
            },
        };
        assert_ne!(c, br, "axis participates in identity");
        let cr = CommandRef {
            command_record_oid: oid(4),
            logical_command_seq: 7,
        };
        assert_eq!(cr, cr);
    }

    #[test]
    fn coordinate_ref_identity_includes_coordinates() {
        let x: ConditionalCoordinateRef<CommitCapsule> =
            ConditionalCoordinateRef::new(oid(5), GraphId(1), BranchId(1));
        let y: ConditionalCoordinateRef<CommitCapsule> =
            ConditionalCoordinateRef::new(oid(5), GraphId(1), BranchId(2));
        assert_ne!(x, y);
        let w: WeakDigest<CommitCapsule> = WeakDigest::new([7; 32]);
        assert_eq!(w, WeakDigest::new([7; 32]));
    }
}
