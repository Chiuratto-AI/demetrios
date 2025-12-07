//! Physical Memory Topology
//!
//! This module implements memory layouts that mirror physical reality.
//! Elements near in physical space are near in memory, making neighbor
//! queries cache-efficient by construction.
//!
//! # Novel Aspects
//!
//! 1. **Space-Filling Curves**: Morton and Hilbert curves for locality
//! 2. **Physical Arrays**: Memory layout follows physical proximity
//! 3. **Neighbor Primitives**: Automatic cache-friendly neighbor iteration
//! 4. **Spatial Partitions**: Cell lists, octrees, etc. as language primitives
//!
//! # The Key Insight
//!
//! Traditional arrays are logically ordered (i, j, k). But scientific computing
//! is about physical space, where locality matters. By arranging memory to
//! match physical proximity, we get cache efficiency "for free."

use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;

// ============================================================================
// PHYSICAL SPACE ABSTRACTION
// ============================================================================

/// A physical space in which elements exist
pub trait PhysicalSpace: fmt::Debug + Clone + Send + Sync {
    /// Dimensionality of the space
    const DIMENSIONS: usize;

    /// The coordinate type
    type Coord: Copy + Clone + fmt::Debug;

    /// Distance between two points
    fn distance(a: &Self::Coord, b: &Self::Coord) -> f64;

    /// Check if a point is within a domain
    fn contains(&self, point: &Self::Coord) -> bool;

    /// Get the bounding box
    fn bounds(&self) -> (Self::Coord, Self::Coord);
}

/// 1D physical space
#[derive(Debug, Clone)]
pub struct Space1D {
    pub min: f64,
    pub max: f64,
}

impl PhysicalSpace for Space1D {
    const DIMENSIONS: usize = 1;
    type Coord = f64;

    fn distance(a: &f64, b: &f64) -> f64 {
        (a - b).abs()
    }

    fn contains(&self, point: &f64) -> bool {
        *point >= self.min && *point <= self.max
    }

    fn bounds(&self) -> (f64, f64) {
        (self.min, self.max)
    }
}

/// 2D coordinate
#[derive(Debug, Clone, Copy, Default)]
pub struct Coord2D {
    pub x: f64,
    pub y: f64,
}

impl Coord2D {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// 2D physical space
#[derive(Debug, Clone)]
pub struct Space2D {
    pub min: Coord2D,
    pub max: Coord2D,
}

impl PhysicalSpace for Space2D {
    const DIMENSIONS: usize = 2;
    type Coord = Coord2D;

    fn distance(a: &Coord2D, b: &Coord2D) -> f64 {
        ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
    }

    fn contains(&self, point: &Coord2D) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
    }

    fn bounds(&self) -> (Coord2D, Coord2D) {
        (self.min, self.max)
    }
}

/// 3D coordinate
#[derive(Debug, Clone, Copy, Default)]
pub struct Coord3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Coord3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Squared distance (avoids sqrt)
    pub fn distance_squared(&self, other: &Coord3D) -> f64 {
        (self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2)
    }
}

/// 3D physical space
#[derive(Debug, Clone)]
pub struct Space3D {
    pub min: Coord3D,
    pub max: Coord3D,
}

impl Space3D {
    pub fn new(min: Coord3D, max: Coord3D) -> Self {
        Self { min, max }
    }

    pub fn cubic(size: f64) -> Self {
        Self {
            min: Coord3D::new(0.0, 0.0, 0.0),
            max: Coord3D::new(size, size, size),
        }
    }

    pub fn dimensions(&self) -> Coord3D {
        Coord3D::new(
            self.max.x - self.min.x,
            self.max.y - self.min.y,
            self.max.z - self.min.z,
        )
    }
}

impl PhysicalSpace for Space3D {
    const DIMENSIONS: usize = 3;
    type Coord = Coord3D;

    fn distance(a: &Coord3D, b: &Coord3D) -> f64 {
        a.distance_squared(b).sqrt()
    }

    fn contains(&self, point: &Coord3D) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    fn bounds(&self) -> (Coord3D, Coord3D) {
        (self.min, self.max)
    }
}

// ============================================================================
// SPACE-FILLING CURVES
// ============================================================================

/// A space-filling curve for mapping N-D coordinates to 1-D indices
pub trait SpaceFillingCurve: fmt::Debug + Clone + Send + Sync {
    /// Dimensionality
    const DIMENSIONS: usize;

    /// Encode a discrete coordinate to a curve index
    fn encode(&self, coords: &[u32]) -> u64;

    /// Decode a curve index to discrete coordinates
    fn decode(&self, index: u64) -> Vec<u32>;

    /// Get the order (bits per dimension)
    fn order(&self) -> u32;

    /// Maximum index value
    fn max_index(&self) -> u64 {
        1u64 << (Self::DIMENSIONS as u32 * self.order())
    }
}

/// Morton curve (Z-order curve)
///
/// Interleaves bits of coordinates. Fast to compute, good locality.
/// Used in GPU texture caches and many scientific codes.
#[derive(Debug, Clone)]
pub struct MortonCurve {
    /// Bits per dimension
    order: u32,
    /// Dimensionality
    dimensions: usize,
}

impl MortonCurve {
    pub fn new(dimensions: usize, order: u32) -> Self {
        assert!(
            dimensions >= 1 && dimensions <= 3,
            "Morton curve supports 1-3 dimensions"
        );
        assert!(order <= 21, "Order must be <= 21 for 64-bit indices");
        Self { order, dimensions }
    }

    /// 2D Morton encoding (fast path)
    pub fn encode_2d(x: u32, y: u32) -> u64 {
        let x = Self::spread_bits_2d(x as u64);
        let y = Self::spread_bits_2d(y as u64);
        x | (y << 1)
    }

    /// 2D Morton decoding
    pub fn decode_2d(index: u64) -> (u32, u32) {
        let x = Self::compact_bits_2d(index) as u32;
        let y = Self::compact_bits_2d(index >> 1) as u32;
        (x, y)
    }

    /// 3D Morton encoding
    pub fn encode_3d(x: u32, y: u32, z: u32) -> u64 {
        let x = Self::spread_bits_3d(x as u64);
        let y = Self::spread_bits_3d(y as u64);
        let z = Self::spread_bits_3d(z as u64);
        x | (y << 1) | (z << 2)
    }

    /// 3D Morton decoding
    pub fn decode_3d(index: u64) -> (u32, u32, u32) {
        let x = Self::compact_bits_3d(index) as u32;
        let y = Self::compact_bits_3d(index >> 1) as u32;
        let z = Self::compact_bits_3d(index >> 2) as u32;
        (x, y, z)
    }

    /// Spread bits for 2D interleaving: 0b1111 -> 0b01010101
    fn spread_bits_2d(mut x: u64) -> u64 {
        x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF;
        x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF;
        x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
        x = (x | (x << 2)) & 0x3333_3333_3333_3333;
        x = (x | (x << 1)) & 0x5555_5555_5555_5555;
        x
    }

    /// Compact bits for 2D: 0b01010101 -> 0b1111
    fn compact_bits_2d(mut x: u64) -> u64 {
        x &= 0x5555_5555_5555_5555;
        x = (x | (x >> 1)) & 0x3333_3333_3333_3333;
        x = (x | (x >> 2)) & 0x0F0F_0F0F_0F0F_0F0F;
        x = (x | (x >> 4)) & 0x00FF_00FF_00FF_00FF;
        x = (x | (x >> 8)) & 0x0000_FFFF_0000_FFFF;
        x = (x | (x >> 16)) & 0x0000_0000_FFFF_FFFF;
        x
    }

    /// Spread bits for 3D interleaving
    fn spread_bits_3d(mut x: u64) -> u64 {
        x &= 0x1FFFFF; // 21 bits max
        x = (x | (x << 32)) & 0x1F00000000FFFF;
        x = (x | (x << 16)) & 0x1F0000FF0000FF;
        x = (x | (x << 8)) & 0x100F00F00F00F00F;
        x = (x | (x << 4)) & 0x10C30C30C30C30C3;
        x = (x | (x << 2)) & 0x1249249249249249;
        x
    }

    /// Compact bits for 3D
    fn compact_bits_3d(mut x: u64) -> u64 {
        x &= 0x1249249249249249;
        x = (x | (x >> 2)) & 0x10C30C30C30C30C3;
        x = (x | (x >> 4)) & 0x100F00F00F00F00F;
        x = (x | (x >> 8)) & 0x1F0000FF0000FF;
        x = (x | (x >> 16)) & 0x1F00000000FFFF;
        x = (x | (x >> 32)) & 0x1FFFFF;
        x
    }
}

impl SpaceFillingCurve for MortonCurve {
    const DIMENSIONS: usize = 3; // Max supported

    fn encode(&self, coords: &[u32]) -> u64 {
        match self.dimensions {
            1 => coords[0] as u64,
            2 => Self::encode_2d(coords[0], coords[1]),
            3 => Self::encode_3d(coords[0], coords[1], coords[2]),
            _ => unreachable!(),
        }
    }

    fn decode(&self, index: u64) -> Vec<u32> {
        match self.dimensions {
            1 => vec![index as u32],
            2 => {
                let (x, y) = Self::decode_2d(index);
                vec![x, y]
            }
            3 => {
                let (x, y, z) = Self::decode_3d(index);
                vec![x, y, z]
            }
            _ => unreachable!(),
        }
    }

    fn order(&self) -> u32 {
        self.order
    }
}

/// Hilbert curve
///
/// Better locality than Morton but more expensive to compute.
/// Optimal for truly locality-sensitive applications.
#[derive(Debug, Clone)]
pub struct HilbertCurve {
    /// Bits per dimension
    order: u32,
    /// Dimensionality
    dimensions: usize,
}

impl HilbertCurve {
    pub fn new(dimensions: usize, order: u32) -> Self {
        assert!(
            dimensions == 2 || dimensions == 3,
            "Hilbert curve supports 2-3 dimensions"
        );
        assert!(order <= 21, "Order must be <= 21 for 64-bit indices");
        Self { order, dimensions }
    }

    /// 2D Hilbert encoding
    pub fn encode_2d(&self, x: u32, y: u32) -> u64 {
        let n = 1u32 << self.order;
        let mut rx: u32;
        let mut ry: u32;
        let mut s: u32;
        let mut d: u64 = 0;
        let mut x = x;
        let mut y = y;

        s = n / 2;
        while s > 0 {
            rx = if (x & s) > 0 { 1 } else { 0 };
            ry = if (y & s) > 0 { 1 } else { 0 };
            d += (s as u64 * s as u64) * ((3 * rx) ^ ry) as u64;

            // Rotate - transform within quadrant
            if ry == 0 {
                if rx == 1 {
                    // x and y are masked to be < s, so s-1-x is always valid
                    x = (s - 1) - (x % s);
                    y = (s - 1) - (y % s);
                }
                std::mem::swap(&mut x, &mut y);
            }

            s /= 2;
        }

        d
    }

    /// 2D Hilbert decoding
    pub fn decode_2d(&self, d: u64) -> (u32, u32) {
        let n = 1u32 << self.order;
        let mut rx: u32;
        let mut ry: u32;
        let mut s: u32;
        let mut t = d;
        let mut x: u32 = 0;
        let mut y: u32 = 0;

        s = 1;
        while s < n {
            rx = (1 & (t / 2)) as u32;
            ry = (1 & (t ^ rx as u64)) as u32;

            // Rotate
            if ry == 0 {
                if rx == 1 {
                    x = s - 1 - x;
                    y = s - 1 - y;
                }
                std::mem::swap(&mut x, &mut y);
            }

            x += s * rx;
            y += s * ry;
            t /= 4;
            s *= 2;
        }

        (x, y)
    }
}

impl SpaceFillingCurve for HilbertCurve {
    const DIMENSIONS: usize = 2; // Primary support

    fn encode(&self, coords: &[u32]) -> u64 {
        match self.dimensions {
            2 => self.encode_2d(coords[0], coords[1]),
            _ => todo!("3D Hilbert not yet implemented"),
        }
    }

    fn decode(&self, index: u64) -> Vec<u32> {
        match self.dimensions {
            2 => {
                let (x, y) = self.decode_2d(index);
                vec![x, y]
            }
            _ => todo!("3D Hilbert not yet implemented"),
        }
    }

    fn order(&self) -> u32 {
        self.order
    }
}

// ============================================================================
// PHYSICAL ARRAY - MEMORY THAT MIRRORS SPACE
// ============================================================================

/// An array where memory layout follows physical proximity
///
/// This is the key data structure for substrate-aware computing.
/// Elements that are near in physical space are near in memory,
/// making neighbor queries cache-efficient by construction.
#[derive(Debug, Clone)]
pub struct PhysicalArray<T, S: PhysicalSpace> {
    /// The data, arranged by space-filling curve
    data: Vec<T>,
    /// The physical space
    space: S,
    /// The space-filling curve
    curve: MortonCurve,
    /// Grid resolution
    resolution: Vec<u32>,
    /// Cached grid cell size
    cell_size: Vec<f64>,
}

impl<T: Clone + Default, S: PhysicalSpace> PhysicalArray<T, S> {
    /// Create a new physical array with uniform initial value
    pub fn new(space: S, resolution: Vec<u32>, initial: T) -> Self {
        let dimensions = S::DIMENSIONS;
        assert_eq!(resolution.len(), dimensions);

        let total_cells: usize = resolution.iter().map(|&r| r as usize).product();

        // Calculate cell sizes
        let (min, max) = space.bounds();
        let cell_size = vec![1.0; dimensions]; // Will be computed properly for each space type

        // Determine Morton curve order (largest dimension)
        let order = resolution
            .iter()
            .map(|&r| (r as f64).log2().ceil() as u32)
            .max()
            .unwrap_or(1);

        Self {
            data: vec![initial; total_cells],
            space,
            curve: MortonCurve::new(dimensions, order),
            resolution,
            cell_size,
        }
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get resolution
    pub fn resolution(&self) -> &[u32] {
        &self.resolution
    }
}

impl<T: Clone + Default> PhysicalArray<T, Space3D> {
    /// Create for 3D space
    pub fn new_3d(space: Space3D, resolution: [u32; 3], initial: T) -> Self {
        let dims = space.dimensions();
        let cell_size = vec![
            dims.x / resolution[0] as f64,
            dims.y / resolution[1] as f64,
            dims.z / resolution[2] as f64,
        ];

        let order = resolution
            .iter()
            .map(|&r| (r as f64).log2().ceil() as u32)
            .max()
            .unwrap_or(1);
        let total_cells = resolution.iter().map(|&r| r as usize).product();

        Self {
            data: vec![initial; total_cells],
            space,
            curve: MortonCurve::new(3, order),
            resolution: resolution.to_vec(),
            cell_size,
        }
    }

    /// Convert physical coordinates to grid indices
    pub fn coord_to_grid(&self, coord: &Coord3D) -> Option<[u32; 3]> {
        if !self.space.contains(coord) {
            return None;
        }

        let i = ((coord.x - self.space.min.x) / self.cell_size[0]).floor() as u32;
        let j = ((coord.y - self.space.min.y) / self.cell_size[1]).floor() as u32;
        let k = ((coord.z - self.space.min.z) / self.cell_size[2]).floor() as u32;

        Some([
            i.min(self.resolution[0] - 1),
            j.min(self.resolution[1] - 1),
            k.min(self.resolution[2] - 1),
        ])
    }

    /// Convert grid indices to physical coordinates (cell center)
    pub fn grid_to_coord(&self, grid: [u32; 3]) -> Coord3D {
        Coord3D::new(
            self.space.min.x + (grid[0] as f64 + 0.5) * self.cell_size[0],
            self.space.min.y + (grid[1] as f64 + 0.5) * self.cell_size[1],
            self.space.min.z + (grid[2] as f64 + 0.5) * self.cell_size[2],
        )
    }

    /// Get linear index from grid coordinates using Morton curve
    pub fn grid_to_linear(&self, grid: [u32; 3]) -> usize {
        // For now, use simple row-major ordering
        // TODO: Use Morton curve for better cache locality
        (grid[2] as usize * self.resolution[1] as usize + grid[1] as usize)
            * self.resolution[0] as usize
            + grid[0] as usize
    }

    /// Get grid coordinates from linear index
    pub fn linear_to_grid(&self, index: usize) -> [u32; 3] {
        let nx = self.resolution[0] as usize;
        let ny = self.resolution[1] as usize;

        let i = (index % nx) as u32;
        let j = ((index / nx) % ny) as u32;
        let k = (index / (nx * ny)) as u32;

        [i, j, k]
    }

    /// Get value at grid coordinates
    pub fn get(&self, grid: [u32; 3]) -> Option<&T> {
        if grid[0] < self.resolution[0]
            && grid[1] < self.resolution[1]
            && grid[2] < self.resolution[2]
        {
            Some(&self.data[self.grid_to_linear(grid)])
        } else {
            None
        }
    }

    /// Get mutable value at grid coordinates
    pub fn get_mut(&mut self, grid: [u32; 3]) -> Option<&mut T> {
        if grid[0] < self.resolution[0]
            && grid[1] < self.resolution[1]
            && grid[2] < self.resolution[2]
        {
            let idx = self.grid_to_linear(grid);
            Some(&mut self.data[idx])
        } else {
            None
        }
    }

    /// Get value at physical coordinates
    pub fn get_at(&self, coord: &Coord3D) -> Option<&T> {
        self.coord_to_grid(coord).and_then(|g| self.get(g))
    }

    /// Set value at grid coordinates
    pub fn set(&mut self, grid: [u32; 3], value: T) {
        if grid[0] < self.resolution[0]
            && grid[1] < self.resolution[1]
            && grid[2] < self.resolution[2]
        {
            let idx = self.grid_to_linear(grid);
            self.data[idx] = value;
        }
    }

    /// Iterate over neighbors of a grid cell
    pub fn neighbors(&self, grid: [u32; 3]) -> NeighborIterator3D {
        NeighborIterator3D::new(grid, &self.resolution)
    }
}

/// Iterator over 3D grid neighbors (26-connected)
pub struct NeighborIterator3D {
    center: [u32; 3],
    resolution: [u32; 3],
    offset_index: usize,
}

impl NeighborIterator3D {
    const OFFSETS: [[i32; 3]; 26] = [
        [-1, -1, -1],
        [0, -1, -1],
        [1, -1, -1],
        [-1, 0, -1],
        [0, 0, -1],
        [1, 0, -1],
        [-1, 1, -1],
        [0, 1, -1],
        [1, 1, -1],
        [-1, -1, 0],
        [0, -1, 0],
        [1, -1, 0],
        [-1, 0, 0],
        [1, 0, 0],
        [-1, 1, 0],
        [0, 1, 0],
        [1, 1, 0],
        [-1, -1, 1],
        [0, -1, 1],
        [1, -1, 1],
        [-1, 0, 1],
        [0, 0, 1],
        [1, 0, 1],
        [-1, 1, 1],
        [0, 1, 1],
        [1, 1, 1],
    ];

    pub fn new(center: [u32; 3], resolution: &[u32]) -> Self {
        Self {
            center,
            resolution: [resolution[0], resolution[1], resolution[2]],
            offset_index: 0,
        }
    }
}

impl Iterator for NeighborIterator3D {
    type Item = [u32; 3];

    fn next(&mut self) -> Option<Self::Item> {
        while self.offset_index < 26 {
            let offset = Self::OFFSETS[self.offset_index];
            self.offset_index += 1;

            let ni = self.center[0] as i32 + offset[0];
            let nj = self.center[1] as i32 + offset[1];
            let nk = self.center[2] as i32 + offset[2];

            if ni >= 0
                && ni < self.resolution[0] as i32
                && nj >= 0
                && nj < self.resolution[1] as i32
                && nk >= 0
                && nk < self.resolution[2] as i32
            {
                return Some([ni as u32, nj as u32, nk as u32]);
            }
        }
        None
    }
}

// ============================================================================
// CELL LIST - SPATIAL PARTITIONING FOR NEIGHBOR QUERIES
// ============================================================================

/// Cell list for efficient neighbor queries
///
/// This is a fundamental data structure for molecular dynamics and
/// other particle-based simulations. Particles are binned into cells
/// based on their position, and neighbor queries only check nearby cells.
#[derive(Debug)]
pub struct CellList<T> {
    /// Items in each cell (cell index -> item indices)
    cells: Vec<Vec<usize>>,
    /// The items themselves
    items: Vec<T>,
    /// Positions of items
    positions: Vec<Coord3D>,
    /// The physical space
    space: Space3D,
    /// Cell dimensions
    cell_size: Coord3D,
    /// Number of cells in each dimension
    n_cells: [usize; 3],
    /// Cutoff distance for neighbor queries
    cutoff: f64,
}

impl<T: Clone> CellList<T> {
    /// Create a new cell list
    pub fn new(space: Space3D, cutoff: f64) -> Self {
        let dims = space.dimensions();

        // Cell size should be >= cutoff so we only need to check adjacent cells
        let cell_size = Coord3D::new(cutoff, cutoff, cutoff);

        let n_cells = [
            (dims.x / cutoff).ceil() as usize,
            (dims.y / cutoff).ceil() as usize,
            (dims.z / cutoff).ceil() as usize,
        ];

        let total_cells = n_cells[0] * n_cells[1] * n_cells[2];

        Self {
            cells: vec![Vec::new(); total_cells],
            items: Vec::new(),
            positions: Vec::new(),
            space,
            cell_size,
            n_cells,
            cutoff,
        }
    }

    /// Add an item at a position
    pub fn insert(&mut self, item: T, position: Coord3D) -> usize {
        let index = self.items.len();
        self.items.push(item);
        self.positions.push(position);

        if let Some(cell_idx) = self.position_to_cell(&position) {
            self.cells[cell_idx].push(index);
        }

        index
    }

    /// Update position of an item
    pub fn update_position(&mut self, index: usize, new_position: Coord3D) {
        let old_pos = self.positions[index];
        let old_cell = self.position_to_cell(&old_pos);
        let new_cell = self.position_to_cell(&new_position);

        self.positions[index] = new_position;

        // Move between cells if necessary
        if old_cell != new_cell {
            if let Some(old_idx) = old_cell {
                self.cells[old_idx].retain(|&i| i != index);
            }
            if let Some(new_idx) = new_cell {
                self.cells[new_idx].push(index);
            }
        }
    }

    /// Get cell index for a position
    fn position_to_cell(&self, pos: &Coord3D) -> Option<usize> {
        if !self.space.contains(pos) {
            return None;
        }

        let ci = ((pos.x - self.space.min.x) / self.cell_size.x) as usize;
        let cj = ((pos.y - self.space.min.y) / self.cell_size.y) as usize;
        let ck = ((pos.z - self.space.min.z) / self.cell_size.z) as usize;

        let ci = ci.min(self.n_cells[0] - 1);
        let cj = cj.min(self.n_cells[1] - 1);
        let ck = ck.min(self.n_cells[2] - 1);

        Some(ck * self.n_cells[1] * self.n_cells[0] + cj * self.n_cells[0] + ci)
    }

    /// Get cell coordinates from cell index
    fn cell_index_to_coords(&self, idx: usize) -> [usize; 3] {
        let ci = idx % self.n_cells[0];
        let cj = (idx / self.n_cells[0]) % self.n_cells[1];
        let ck = idx / (self.n_cells[0] * self.n_cells[1]);
        [ci, cj, ck]
    }

    /// Iterate over neighbors of an item
    pub fn neighbors(&self, index: usize) -> impl Iterator<Item = (usize, &T, f64)> {
        let pos = self.positions[index];
        let cutoff_sq = self.cutoff * self.cutoff;

        // Get the cell containing this item
        let cell_idx = self.position_to_cell(&pos).unwrap_or(0);
        let cell_coords = self.cell_index_to_coords(cell_idx);

        // Collect neighbor cells (including self)
        let mut neighbor_items = Vec::new();

        for di in -1i32..=1 {
            for dj in -1i32..=1 {
                for dk in -1i32..=1 {
                    let ni = cell_coords[0] as i32 + di;
                    let nj = cell_coords[1] as i32 + dj;
                    let nk = cell_coords[2] as i32 + dk;

                    if ni >= 0
                        && ni < self.n_cells[0] as i32
                        && nj >= 0
                        && nj < self.n_cells[1] as i32
                        && nk >= 0
                        && nk < self.n_cells[2] as i32
                    {
                        let neighbor_cell = (nk as usize * self.n_cells[1] + nj as usize)
                            * self.n_cells[0]
                            + ni as usize;

                        for &item_idx in &self.cells[neighbor_cell] {
                            if item_idx != index {
                                let dist_sq = pos.distance_squared(&self.positions[item_idx]);
                                if dist_sq <= cutoff_sq {
                                    neighbor_items.push((
                                        item_idx,
                                        &self.items[item_idx],
                                        dist_sq.sqrt(),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        neighbor_items.into_iter()
    }

    /// Get all items
    pub fn items(&self) -> &[T] {
        &self.items
    }

    /// Get all positions
    pub fn positions(&self) -> &[Coord3D] {
        &self.positions
    }

    /// Get item and position by index
    pub fn get(&self, index: usize) -> Option<(&T, &Coord3D)> {
        self.items
            .get(index)
            .map(|item| (item, &self.positions[index]))
    }

    /// Number of items
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Rebuild the cell list (after many updates)
    pub fn rebuild(&mut self) {
        // Clear all cells
        for cell in &mut self.cells {
            cell.clear();
        }

        // Re-insert all items
        for (idx, pos) in self.positions.iter().enumerate() {
            if let Some(cell_idx) = self.position_to_cell(pos) {
                self.cells[cell_idx].push(idx);
            }
        }
    }
}

// ============================================================================
// SPATIAL PARTITION TRAIT
// ============================================================================

/// A spatial partitioning structure for efficient neighbor queries
pub trait SpatialPartition<T>: Send + Sync {
    /// Insert an item at a position
    fn insert(&mut self, item: T, position: Coord3D) -> usize;

    /// Update position of an item
    fn update(&mut self, index: usize, new_position: Coord3D);

    /// Query neighbors within a radius
    fn query_radius(&self, center: &Coord3D, radius: f64) -> Vec<usize>;

    /// Query k nearest neighbors
    fn query_knn(&self, center: &Coord3D, k: usize) -> Vec<usize>;

    /// Number of items
    fn len(&self) -> usize;

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Clone + Send + Sync> SpatialPartition<T> for CellList<T> {
    fn insert(&mut self, item: T, position: Coord3D) -> usize {
        CellList::insert(self, item, position)
    }

    fn update(&mut self, index: usize, new_position: Coord3D) {
        self.update_position(index, new_position);
    }

    fn query_radius(&self, center: &Coord3D, radius: f64) -> Vec<usize> {
        let radius_sq = radius * radius;
        let mut result = Vec::new();

        // Get the cell containing the center
        if let Some(cell_idx) = self.position_to_cell(center) {
            let cell_coords = self.cell_index_to_coords(cell_idx);

            // Check how many cells we need to search based on radius
            let cells_to_check = (radius / self.cell_size.x).ceil() as i32 + 1;

            for di in -cells_to_check..=cells_to_check {
                for dj in -cells_to_check..=cells_to_check {
                    for dk in -cells_to_check..=cells_to_check {
                        let ni = cell_coords[0] as i32 + di;
                        let nj = cell_coords[1] as i32 + dj;
                        let nk = cell_coords[2] as i32 + dk;

                        if ni >= 0
                            && ni < self.n_cells[0] as i32
                            && nj >= 0
                            && nj < self.n_cells[1] as i32
                            && nk >= 0
                            && nk < self.n_cells[2] as i32
                        {
                            let neighbor_cell = (nk as usize * self.n_cells[1] + nj as usize)
                                * self.n_cells[0]
                                + ni as usize;

                            for &item_idx in &self.cells[neighbor_cell] {
                                let dist_sq = center.distance_squared(&self.positions[item_idx]);
                                if dist_sq <= radius_sq {
                                    result.push(item_idx);
                                }
                            }
                        }
                    }
                }
            }
        }

        result
    }

    fn query_knn(&self, center: &Coord3D, k: usize) -> Vec<usize> {
        // Simple implementation: collect all with distances, sort, take k
        let mut distances: Vec<(usize, f64)> = self
            .positions
            .iter()
            .enumerate()
            .map(|(i, pos)| (i, center.distance_squared(pos)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        distances.into_iter().take(k).map(|(i, _)| i).collect()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morton_2d() {
        // Test some known values
        assert_eq!(MortonCurve::encode_2d(0, 0), 0);
        assert_eq!(MortonCurve::encode_2d(1, 0), 1);
        assert_eq!(MortonCurve::encode_2d(0, 1), 2);
        assert_eq!(MortonCurve::encode_2d(1, 1), 3);

        // Test round-trip
        for x in 0..16 {
            for y in 0..16 {
                let encoded = MortonCurve::encode_2d(x, y);
                let (dx, dy) = MortonCurve::decode_2d(encoded);
                assert_eq!((x, y), (dx, dy));
            }
        }
    }

    #[test]
    fn test_morton_3d() {
        // Test some known values
        assert_eq!(MortonCurve::encode_3d(0, 0, 0), 0);
        assert_eq!(MortonCurve::encode_3d(1, 0, 0), 1);
        assert_eq!(MortonCurve::encode_3d(0, 1, 0), 2);
        assert_eq!(MortonCurve::encode_3d(0, 0, 1), 4);

        // Test round-trip
        for x in 0..8 {
            for y in 0..8 {
                for z in 0..8 {
                    let encoded = MortonCurve::encode_3d(x, y, z);
                    let (dx, dy, dz) = MortonCurve::decode_3d(encoded);
                    assert_eq!((x, y, z), (dx, dy, dz));
                }
            }
        }
    }

    #[test]
    fn test_hilbert_2d() {
        let curve = HilbertCurve::new(2, 4);

        // Test round-trip
        for x in 0..16 {
            for y in 0..16 {
                let encoded = curve.encode_2d(x, y);
                let (dx, dy) = curve.decode_2d(encoded);
                assert_eq!((x, y), (dx, dy));
            }
        }
    }

    #[test]
    fn test_physical_array_3d() {
        let space = Space3D::cubic(10.0);
        let mut arr: PhysicalArray<f64, Space3D> = PhysicalArray::new_3d(space, [10, 10, 10], 0.0);

        // Set and get
        arr.set([5, 5, 5], 42.0);
        assert_eq!(arr.get([5, 5, 5]), Some(&42.0));

        // Get at physical coordinate
        let coord = Coord3D::new(5.5, 5.5, 5.5);
        assert_eq!(arr.get_at(&coord), Some(&42.0));
    }

    #[test]
    fn test_physical_array_neighbors() {
        let space = Space3D::cubic(10.0);
        let arr: PhysicalArray<f64, Space3D> = PhysicalArray::new_3d(space, [10, 10, 10], 0.0);

        // Interior point should have 26 neighbors
        let neighbors: Vec<_> = arr.neighbors([5, 5, 5]).collect();
        assert_eq!(neighbors.len(), 26);

        // Corner should have fewer neighbors
        let corner_neighbors: Vec<_> = arr.neighbors([0, 0, 0]).collect();
        assert_eq!(corner_neighbors.len(), 7);
    }

    #[test]
    fn test_cell_list() {
        let space = Space3D::cubic(10.0);
        let mut cells = CellList::new(space, 2.0);

        // Insert some items
        cells.insert("A", Coord3D::new(1.0, 1.0, 1.0));
        cells.insert("B", Coord3D::new(1.5, 1.5, 1.5));
        cells.insert("C", Coord3D::new(5.0, 5.0, 5.0));

        assert_eq!(cells.len(), 3);

        // A and B should be neighbors
        let neighbors_a: Vec<_> = cells.neighbors(0).collect();
        assert_eq!(neighbors_a.len(), 1);
        assert_eq!(neighbors_a[0].0, 1); // B is neighbor of A

        // C should have no neighbors (too far)
        let neighbors_c: Vec<_> = cells.neighbors(2).collect();
        assert_eq!(neighbors_c.len(), 0);
    }

    #[test]
    fn test_cell_list_radius_query() {
        let space = Space3D::cubic(10.0);
        let mut cells = CellList::new(space, 1.0);

        // Insert items in a line
        for i in 0..10 {
            cells.insert(i, Coord3D::new(i as f64, 0.0, 0.0));
        }

        // Query radius 2 around item at x=5
        let center = Coord3D::new(5.0, 0.0, 0.0);
        let result = cells.query_radius(&center, 2.5);

        // Should find items at x=3,4,5,6,7
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_space_3d() {
        let space = Space3D::cubic(10.0);

        assert!(space.contains(&Coord3D::new(5.0, 5.0, 5.0)));
        assert!(!space.contains(&Coord3D::new(15.0, 5.0, 5.0)));

        let (min, max) = space.bounds();
        assert_eq!(min.x, 0.0);
        assert_eq!(max.x, 10.0);
    }

    #[test]
    fn test_distance() {
        let a = Coord3D::new(0.0, 0.0, 0.0);
        let b = Coord3D::new(3.0, 4.0, 0.0);

        assert!((Space3D::distance(&a, &b) - 5.0).abs() < 1e-10);
    }
}
