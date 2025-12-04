//! Parallel compilation utilities

pub mod lexer;
pub mod scheduler;
pub mod type_check;

pub use lexer::{ParallelLexConfig, ParallelLexStats, ParallelLexer};
pub use scheduler::{SchedulerStats, Task, WorkStealingScheduler};
pub use type_check::{DependencyGraph, NodeKey, ParallelTypeChecker};
