//! Dynamic-window cryptanalysis: Rust orchestration and FP64 CUDA kernels.
//! See docs/architecture.md for the transition model and module boundaries.
pub mod affine;
pub mod compressed;
pub mod cuda;
pub mod engine;
pub mod model;
pub mod multi;
pub mod posterior;
pub mod refine;
pub mod run;
pub mod union;
