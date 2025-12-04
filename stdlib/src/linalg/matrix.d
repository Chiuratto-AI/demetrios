//! Dense matrix library with BLAS/LAPACK backend

use std::mem::{alloc, dealloc, copy}
use std::iter::Iterator
use std::ops::{Add, Sub, Mul, Div, Index, IndexMut}
use units::{Dimensionless}

/// Memory layout for matrices
pub enum Layout {
    /// Row-major (C-style): A[i,j] at offset i*ncols + j
    RowMajor,

    /// Column-major (Fortran-style): A[i,j] at offset j*nrows + i
    ColMajor,
}

/// A dense matrix with configurable element type and layout
pub struct Matrix<T, const L: Layout = Layout::RowMajor> {
    /// Raw data storage
    data: own Vec<T>,

    /// Number of rows
    nrows: usize,

    /// Number of columns
    ncols: usize,

    /// Leading dimension (stride between rows/cols)
    ld: usize,
}

impl<T: Clone + Default, const L: Layout> Matrix<T, L> {
    /// Create a new matrix filled with default values
    pub fn new(nrows: usize, ncols: usize) -> Self {
        let size = nrows * ncols;
        let mut data = Vec::with_capacity(size);
        data.resize(size, T::default());

        let ld = match L {
            Layout::RowMajor => ncols,
            Layout::ColMajor => nrows,
        };

        Matrix { data, nrows, ncols, ld }
    }

    /// Create from raw data
    pub fn from_data(data: Vec<T>, nrows: usize, ncols: usize) -> Self {
        assert!(data.len() >= nrows * ncols, "insufficient data");

        let ld = match L {
            Layout::RowMajor => ncols,
            Layout::ColMajor => nrows,
        };

        Matrix { data, nrows, ncols, ld }
    }

    /// Create a zero matrix
    pub fn zeros(nrows: usize, ncols: usize) -> Self
    where T: num::Zero
    {
        let mut m = Self::new(nrows, ncols);
        for i in 0..m.data.len() {
            m.data[i] = T::zero();
        }
        m
    }

    /// Create an identity matrix
    pub fn eye(n: usize) -> Self
    where T: num::Zero + num::One
    {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = T::one();
        }
        m
    }

    /// Create a diagonal matrix
    pub fn diag(values: &[T]) -> Self
    where T: num::Zero
    {
        let n = values.len();
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = values[i].clone();
        }
        m
    }

    /// Create from a 2D nested array
    pub fn from_nested(data: &[[T]]) -> Self {
        let nrows = data.len();
        let ncols = if nrows > 0 { data[0].len() } else { 0 };

        let mut m = Self::new(nrows, ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                m[(i, j)] = data[i][j].clone();
            }
        }
        m
    }

    /// Number of rows
    pub fn nrows(&self) -> usize { self.nrows }

    /// Number of columns
    pub fn ncols(&self) -> usize { self.ncols }

    /// Shape as (rows, cols)
    pub fn shape(&self) -> (usize, usize) { (self.nrows, self.ncols) }

    /// Total number of elements
    pub fn len(&self) -> usize { self.nrows * self.ncols }

    /// Is this a square matrix?
    pub fn is_square(&self) -> bool { self.nrows == self.ncols }

    /// Raw data pointer (for BLAS)
    pub fn as_ptr(&self) -> *const T { self.data.as_ptr() }

    /// Mutable raw data pointer
    pub fn as_mut_ptr(&!self) -> *mut T { self.data.as_mut_ptr() }

    /// Leading dimension
    pub fn ld(&self) -> usize { self.ld }

    /// Convert offset to linear index
    fn linear_index(&self, row: usize, col: usize) -> usize {
        match L {
            Layout::RowMajor => row * self.ld + col,
            Layout::ColMajor => col * self.ld + row,
        }
    }

    /// Get a row as a vector view
    pub fn row(&self, i: usize) -> VectorView<T> {
        assert!(i < self.nrows);
        let start = self.linear_index(i, 0);
        let stride = match L {
            Layout::RowMajor => 1,
            Layout::ColMajor => self.ld,
        };
        VectorView::new(&self.data[start..], self.ncols, stride)
    }

    /// Get a column as a vector view
    pub fn col(&self, j: usize) -> VectorView<T> {
        assert!(j < self.ncols);
        let start = self.linear_index(0, j);
        let stride = match L {
            Layout::RowMajor => self.ld,
            Layout::ColMajor => 1,
        };
        VectorView::new(&self.data[start..], self.nrows, stride)
    }

    /// Transpose (returns a view with swapped dimensions)
    pub fn t(&self) -> MatrixView<T, {L.transpose()}> {
        MatrixView {
            data: &self.data,
            nrows: self.ncols,
            ncols: self.nrows,
            ld: self.ld,
            row_stride: match L {
                Layout::RowMajor => 1,
                Layout::ColMajor => self.ld,
            },
            col_stride: match L {
                Layout::RowMajor => self.ld,
                Layout::ColMajor => 1,
            },
        }
    }

    /// Deep copy
    pub fn clone(&self) -> Self {
        Matrix {
            data: self.data.clone(),
            nrows: self.nrows,
            ncols: self.ncols,
            ld: self.ld,
        }
    }

    /// Reshape (must preserve total elements)
    pub fn reshape(&self, new_rows: usize, new_cols: usize) -> Self {
        assert!(new_rows * new_cols == self.nrows * self.ncols,
                "reshape must preserve total elements");

        Matrix {
            data: self.data.clone(),
            nrows: new_rows,
            ncols: new_cols,
            ld: match L {
                Layout::RowMajor => new_cols,
                Layout::ColMajor => new_rows,
            },
        }
    }

    /// Flatten to 1D vector
    pub fn flatten(&self) -> Vector<T> {
        Vector::from_data(self.data.clone())
    }

    /// Apply element-wise function
    pub fn map<U, F>(&self, f: F) -> Matrix<U, L>
    where
        F: Fn(&T) -> U,
        U: Clone + Default,
    {
        let new_data: Vec<U> = self.data.iter().map(f).collect();
        Matrix::from_data(new_data, self.nrows, self.ncols)
    }

    /// Reduce along axis
    pub fn reduce<F>(&self, axis: usize, f: F) -> Vector<T>
    where F: Fn(&T, &T) -> T
    {
        match axis {
            0 => {
                // Reduce rows -> vector of length ncols
                let mut result = Vector::new(self.ncols);
                for j in 0..self.ncols {
                    result[j] = self[(0, j)].clone();
                    for i in 1..self.nrows {
                        result[j] = f(&result[j], &self[(i, j)]);
                    }
                }
                result
            }
            1 => {
                // Reduce cols -> vector of length nrows
                let mut result = Vector::new(self.nrows);
                for i in 0..self.nrows {
                    result[i] = self[(i, 0)].clone();
                    for j in 1..self.ncols {
                        result[i] = f(&result[i], &self[(i, j)]);
                    }
                }
                result
            }
            _ => panic!("axis must be 0 or 1"),
        }
    }

    /// Sum all elements
    pub fn sum(&self) -> T
    where T: num::Zero + Add<Output = T>
    {
        self.data.iter().fold(T::zero(), |acc, x| acc + x.clone())
    }

    /// Product of all elements
    pub fn prod(&self) -> T
    where T: num::One + Mul<Output = T>
    {
        self.data.iter().fold(T::one(), |acc, x| acc * x.clone())
    }

    /// Mean of all elements
    pub fn mean(&self) -> T
    where T: num::Zero + Add<Output = T> + Div<Output = T> + From<usize>
    {
        self.sum() / T::from(self.len())
    }

    /// Frobenius norm
    pub fn norm_fro(&self) -> f64
    where T: Into<f64> + Clone
    {
        let sum_sq: f64 = self.data.iter()
            .map(|x| {
                let v: f64 = x.clone().into();
                v * v
            })
            .sum();
        sum_sq.sqrt()
    }
}

/// Index operator for matrix
impl<T, const L: Layout> Index<(usize, usize)> for Matrix<T, L> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &T {
        assert!(row < self.nrows && col < self.ncols, "index out of bounds");
        &self.data[self.linear_index(row, col)]
    }
}

/// Mutable index operator
impl<T, const L: Layout> IndexMut<(usize, usize)> for Matrix<T, L> {
    fn index_mut(&!self, (row, col): (usize, usize)) -> &!T {
        assert!(row < self.nrows && col < self.ncols, "index out of bounds");
        let idx = self.linear_index(row, col);
        &!self.data[idx]
    }
}

/// Dense vector (special case of Matrix with 1 column)
pub type Vector<T> = Matrix<T, Layout::ColMajor>;

impl<T: Clone + Default> Vector<T> {
    /// Create a vector of given length
    pub fn new(len: usize) -> Self {
        Matrix::new(len, 1)
    }

    /// Create from slice
    pub fn from_slice(data: &[T]) -> Self {
        Matrix::from_data(data.to_vec(), data.len(), 1)
    }

    /// Vector length
    pub fn len(&self) -> usize { self.nrows }

    /// Dot product
    pub fn dot(&self, other: &Vector<T>) -> T
    where T: num::Zero + Add<Output = T> + Mul<Output = T>
    {
        assert!(self.len() == other.len(), "vectors must have same length");

        let mut sum = T::zero();
        for i in 0..self.len() {
            sum = sum + self[i].clone() * other[i].clone();
        }
        sum
    }

    /// L2 norm
    pub fn norm(&self) -> f64
    where T: Into<f64> + Clone
    {
        self.norm_fro()
    }

    /// Normalize to unit length
    pub fn normalize(&self) -> Self
    where T: Into<f64> + From<f64> + Clone + Default + Div<Output = T>
    {
        let n = self.norm();
        self.map(|x| T::from(x.clone().into() / n))
    }

    /// Cross product (3D only)
    pub fn cross(&self, other: &Vector<T>) -> Self
    where T: Clone + Default + Sub<Output = T> + Mul<Output = T>
    {
        assert!(self.len() == 3 && other.len() == 3, "cross product requires 3D vectors");

        let mut result = Vector::new(3);
        result[0] = self[1].clone() * other[2].clone() - self[2].clone() * other[1].clone();
        result[1] = self[2].clone() * other[0].clone() - self[0].clone() * other[2].clone();
        result[2] = self[0].clone() * other[1].clone() - self[1].clone() * other[0].clone();
        result
    }
}

/// Matrix view (non-owning reference)
pub struct MatrixView<'a, T, const L: Layout = Layout::RowMajor> {
    data: &'a [T],
    nrows: usize,
    ncols: usize,
    ld: usize,
    row_stride: usize,
    col_stride: usize,
}

/// Vector view (non-owning strided reference)
pub struct VectorView<'a, T> {
    data: &'a [T],
    len: usize,
    stride: usize,
}

impl<'a, T> VectorView<'a, T> {
    pub fn new(data: &'a [T], len: usize, stride: usize) -> Self {
        VectorView { data, len, stride }
    }

    pub fn len(&self) -> usize { self.len }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.len).map(move |i| &self.data[i * self.stride])
    }
}

impl<'a, T> Index<usize> for VectorView<'a, T> {
    type Output = T;

    fn index(&self, i: usize) -> &T {
        &self.data[i * self.stride]
    }
}
