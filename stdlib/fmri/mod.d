// fMRI Analysis Module
// Comprehensive neuroimaging analysis for resting-state and task fMRI
//
// Submodules:
// - nifti: NIfTI file format support, voxel access
// - preprocess: Motion correction, temporal filtering, detrending
// - connectivity: Functional connectivity, Fisher-z, network metrics

pub mod nifti;
pub mod preprocess;
pub mod connectivity;
