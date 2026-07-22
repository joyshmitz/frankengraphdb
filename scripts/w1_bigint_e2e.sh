#!/usr/bin/env bash
# End-to-end proof for the first W1 engine crate (fgdb-bigint-kernel-5win).
#
# The evidence directory is intentionally retained. Repository policy forbids
# automated deletion, and the deterministic transcripts are useful for replay.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EVIDENCE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fgdb-bigint-e2e.XXXXXX")"
FIRST="$EVIDENCE_DIR/first.txt"
SECOND="$EVIDENCE_DIR/second.txt"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$EVIDENCE_DIR/target}"

echo "==> verify fgdb-bigint has no normal dependencies"
test "$(cargo tree -p fgdb-bigint --edges normal --depth 1 --prefix none | wc -l)" -eq 1

echo "==> run the complete fgdb-bigint test target twice"
cargo test -p fgdb-bigint --all-targets
cargo test -p fgdb-bigint --all-targets

echo "==> reproduce the arithmetic transcript twice"
cargo run --quiet -p fgdb-bigint --example deterministic_transcript >"$FIRST"
cargo run --quiet -p fgdb-bigint --example deterministic_transcript >"$SECOND"
cmp "$FIRST" "$SECOND"

echo "==> transcript sha256"
sha256sum "$FIRST"
echo "fgdb-bigint E2E GREEN; retained deterministic evidence: $EVIDENCE_DIR"
