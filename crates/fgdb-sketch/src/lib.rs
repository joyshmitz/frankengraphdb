//! Deterministic graph-statistics sketches.
//!
//! Sketches are advisory summaries, never authoritative graph state. Every
//! implementation fixes its merge and deletion behavior explicitly and exposes
//! a canonical logical state for registry-generated durable encoders.

#![forbid(unsafe_code)]

pub mod count_min;
pub mod degree_histogram;
