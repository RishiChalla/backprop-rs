use std::fmt::Display;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape(Vec<usize>);

impl TensorShape {
    /// The number of dimensions in this tensor's shape.
    pub fn num_dims(&self) -> usize { self.0.len() }
    /// Gets the slice of the shape
    pub fn get(&self) -> &[usize] { &self.0 }
    /// Gets the number of elements in a tensor of this shape
    pub fn num_elements(&self) -> usize { self.0.iter().product() }

    /// Gets a slice of the last two dimensions of this tensor. Leading dimensions are treated as batch dimensions.
    /// If this is a 1-dimensional vector, the returned dimensions will have length 1 instead of 2.
    pub fn last_two_dims(&self) -> &[usize] {
        if self.0.len() <= 2 {
            &self.0
        } else {
            &self.0[self.0.len() - 2..]
        }
    }

    /// Gets the batch dimensions of this shape. This is all dimensions except the last 2.
    /// If dimensions count is less than or equal to 2, an empty slice is returned.
    pub fn batch_dims(&self) -> &[usize] {
        if self.0.len() <= 2 {
            &[]
        } else {
            &self.0[..self.0.len() - 2]
        }
    }

    /// Checks if an operation where self is left, and other is right, matches for same size operations (addition/subtraction/etc).
    /// Returns true for operations where batching would succeed, even if exact shape is unequal.
    pub fn matches_same_size_op(&self, other: &TensorShape) -> bool {
        self.matches_batch_size_op(other) && self.last_two_dims() == other.last_two_dims()
    }

    /// Checks if a matrix multiplication operation where self is left, and other is right, matches the shape requirements.
    /// Returns true for operations where batching would succeed, even if number of dimensions is unequal.
    pub fn matches_mul_size_op(&self, other: &TensorShape) -> bool {
        let left_dims = self.last_two_dims();
        let right_dims = other.last_two_dims();
        // We only support matrix multiplication. For vector to matrix multiplications, just add an extra dimensions on your tensor.
        if left_dims.len() != 2 || right_dims.len() != 2 { return false; }
        // Matrix multiplication requires left width = right height
        self.matches_batch_size_op(other) && left_dims[1] == right_dims[0]
    }

    /// Checks if an operation where self is left, and other is right, supports batched operations.
    /// We only support batch operations where the left size has more dimensions than the right size,
    /// OR the batch shape of left and right are fully equal.
    /// This means broadcasting is only supported from left to right.
    pub fn matches_batch_size_op(&self, other: &TensorShape) -> bool {
        let left_batch_dims = self.batch_dims();
        let right_batch_dims = other.batch_dims();
        // No batching
        if left_batch_dims.is_empty() && right_batch_dims.is_empty() { return true; }
        // Same size operation
        if left_batch_dims == right_batch_dims { return true; }
        // Broadcasting is only supported with batch dimensions on left side, right side may only have 2 dimensions.
        right_batch_dims.is_empty()
    }
}

impl Display for TensorShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
