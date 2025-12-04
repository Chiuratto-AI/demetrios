//! Salsa-inspired query system for incremental compilation.

pub mod database;
pub mod queries;

pub use database::{Durability, QueryDatabase, QueryKey, Revision};
pub use queries::*;
