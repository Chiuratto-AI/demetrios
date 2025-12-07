//! Code Generation
//!
//! Provides backends for generating executable code from GPU IR.

pub mod cpu;
pub mod ptx;

pub use cpu::{CpuInterpreter, CpuSimError, ThreadContext, Value};
pub use ptx::{PtxError, PtxGenerator};
