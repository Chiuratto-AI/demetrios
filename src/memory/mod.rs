//! Memory optimization utilities: arenas and string interning.

pub mod arena;
pub mod intern;

pub use arena::{Arena, TypedArena};
pub use intern::{get, intern, Interner, Symbol};
