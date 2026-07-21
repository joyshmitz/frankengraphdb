#!/usr/bin/env bash
# =============================================================================
# g0_spine_e2e.sh — end-to-end proof of the twenty-invariant spine
# (bead fgdb-g0-invariant-spine-tmm)
#
# Verifies the materialized registry: the twenty-ID table hash, resolution of
# every checker and negative-test symbol (stub-registered pre-Genesis), the
# activation closure for the sample capability manifest, and both negative
# fixtures (a twenty-first ID; a reachable-but-inactive clause), asserting
# each fails naming the exact clause. JSONL evidence retained for later gates
# to diff activation drift against this baseline.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${G0_E2E_WORKDIR:-$(mktemp -d)}"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
BIN="$TARGET_DIR/debug/registry-check"
PASS=0
FAIL=0

log() { printf '[g0-spine-e2e] %s\n' "$*"; }
ok()  { PASS=$((PASS + 1)); log "PASS: $*"; }
die() { FAIL=$((FAIL + 1)); log "FAIL: $*"; }

log "work directory: $WORK"
mkdir -p "$WORK"

log "building registry-check"
(cd "$ROOT" && cargo build -p registry-check --quiet)
[ -x "$BIN" ] || { log "registry-check binary missing at $BIN"; exit 2; }

# --- Phase 1: materialized spine passes validate + hash + closure ------------
log "phase 1: materialized spine (validate + hash + closure baseline)"
if "$BIN" all --root "$ROOT" >"$WORK/spine-baseline.jsonl" 2>"$WORK/spine-baseline.err"; then
  ok "materialized spine passes validate/hash/lint/closure"
else
  die "materialized spine failed (see $WORK/spine-baseline.jsonl)"
fi
CLAUSES=$(grep -c '"event":"clause_checked"' "$WORK/spine-baseline.jsonl" || true)
if [ "$CLAUSES" -ge 20 ]; then
  ok "clause_checked events for all materialized clauses ($CLAUSES >= 20)"
else
  die "expected >= 20 clause_checked events, found $CLAUSES"
fi
grep -q '"event":"hash_checked".*"outcome":"pass"' "$WORK/spine-baseline.jsonl" \
  && ok "twenty-ID table hash verified" \
  || die "twenty-ID table hash not verified"
grep -q '"event":"closure_computed".*"absent":0.*"outcome":"pass"' "$WORK/spine-baseline.jsonl" \
  && ok "pre-Genesis sample-manifest closure satisfied (no reachable stubs)" \
  || die "baseline closure not satisfied"
if grep -q '"code":"missing_checker"' "$WORK/spine-baseline.jsonl"; then
  die "unresolvable checker/negative-test symbol on the shipped spine"
else
  ok "every checker and negative-test symbol resolves (stub-registered)"
fi

# --- Phase 2: twenty-first ID fails naming the exact row ---------------------
log "phase 2: planted twenty-first invariant ID"
SPINE="$WORK/spine-stage"
mkdir -p "$SPINE/registries"
cp "$ROOT"/registries/*.toml "$SPINE/registries/"
cat >> "$SPINE/registries/invariants.toml" <<'EOF'

[[invariant]]
id = "FG-INV-21"
title = "planted illegal twenty-first row"
EOF
if "$BIN" validate --root "$SPINE" >"$WORK/spine-neg-21.jsonl" 2>/dev/null; then
  die "validate passed despite twenty-first ID"
else
  ok "validate failed as required on twenty-first ID"
fi
grep -q '"code":"twenty_id_violation".*FG-INV-21' "$WORK/spine-neg-21.jsonl" \
  && ok "violation names FG-INV-21 exactly" \
  || die "twenty_id_violation missing FG-INV-21 (see $WORK/spine-neg-21.jsonl)"

# --- Phase 3: reachable-but-inactive clause forces the capability off --------
log "phase 3: capability manifest enabling a stub-guarded feature"
cat > "$WORK/hot-manifest.toml" <<'EOF'
schema_version = 1
[manifest]
name = "e2e-hot"
description = "enables mvcc-visibility before its checker is live"
features = ["mvcc-visibility"]
postures = []
roles = []
EOF
if "$BIN" closure --root "$ROOT" --manifest "$WORK/hot-manifest.toml" \
     >"$WORK/spine-closure-hot.jsonl" 2>/dev/null; then
  die "closure passed despite reachable stub clause"
else
  ok "closure failed as required on reachable stub clause"
fi
grep -q '"event":"closure_computed".*FG-INV-04.core' "$WORK/spine-closure-hot.jsonl" \
  && ok "closure names the exact absent clause (FG-INV-04.core)" \
  || die "absent clause not named (see $WORK/spine-closure-hot.jsonl)"
grep -q '"event":"capability_absent","capability":"mvcc-visibility"' "$WORK/spine-closure-hot.jsonl" \
  && ok "capability_absent names mvcc-visibility with its clauses" \
  || die "capability_absent event missing"

# --- Verdict -----------------------------------------------------------------
log "evidence: $WORK/{spine-baseline,spine-neg-21,spine-closure-hot}.jsonl"
log "result: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
log "G0 spine e2e: ALL GREEN"
