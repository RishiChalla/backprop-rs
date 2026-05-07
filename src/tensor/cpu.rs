use crate::{ConstructableTensor, Tensor, TensorOpError, TensorOperation, TensorOutput, TensorShape};


/// Tensor stored on the CPU, with no vectorization (no CUDA, no SIMD). A naive implementation with wide compatibility.
#[derive(Debug, Clone, PartialEq)]
pub struct CPUTensor {
    /// Tensor data. Data is stored row-wise -
    /// At one dim: [...row]
    /// At 2 dims: [...row1, ...row2, ...row3]
    /// At 3 dims: [...batch1row1, ...batch1row2, ...batch1row3, ...batch2row1, ...batch2row2, ...batch2row3]
    /// ...
    data: Vec<f32>,
    /// Shape of the tensor
    shape: TensorShape,
}

impl ConstructableTensor for CPUTensor {
    /// Creates a zero-initialized tensor of the given shape.
    fn new(shape: TensorShape) -> Self {
        Self {
            data: vec![0.0; shape.num_elements()],
            shape,
        }
    }

    /// Creates an initialized tensor of the given shape filled with the given scalar.
    fn from_scalar(shape: TensorShape, scalar: f32) -> Self {
        Self {
            data: vec![scalar; shape.num_elements()],
            shape,
        }
    }
}

impl CPUTensor {
    /// Creates a tensor from flat row-major data when it exactly fills the shape.
    pub fn from_data(shape: TensorShape, data: Vec<f32>) -> Option<Self> {
        if data.len() != shape.num_elements() { return None; }

        Some(Self { data, shape })
    }

    /// Checks whether shape and values match within an implementation-specific tolerance.
    pub fn eq_epsilon(&self, other: &Self, epsilon: f32) -> bool {
        self.shape == other.shape
            && self.data.len() == other.data.len()
            && self.data.iter().zip(other.data.iter()).all(|(a, b)| (a - b).abs() <= epsilon)
    }
}

impl Tensor for CPUTensor {
    // Single-var operations
    fn relu(&self) -> Result<Self, TensorOpError> {
        Ok(Self {
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| x.max(0.0)).collect(),
        })
    }
    fn softmax(&self) -> Result<Self, TensorOpError> {
        let exp_sum = self.data.iter().cloned().map(f32::exp).sum::<f32>();
        Ok(Self {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| f32::exp(x) / exp_sum).collect(),
        })
    }
    fn neg(&self) -> Result<Self, TensorOpError> {
        self.mul_scalar(-1.0).map_err(|err| err.with_op(TensorOperation::Neg))
    }

    // Multi-var operations
    fn add(&self, other: &Self) -> Result<Self, TensorOpError> {
        if !self.shape.matches_same_size_op(&other.shape) {
            return Err(TensorOpError::shape_mismatch(TensorOperation::Add, &self.shape, &other.shape));
        }

        let right_count = other.shape.num_elements();

        Ok(Self {
            shape: self.shape.clone(),
            data: self.data.iter().enumerate().map(|(idx, a)| a + other.data[idx % right_count]).collect(),
        })
    }
    fn sub(&self, other: &Self) -> Result<Self, TensorOpError> {
        let neg = other.neg().map_err(|err| err.with_op(TensorOperation::Sub))?;
        self.add(&neg).map_err(|err| err.with_op(TensorOperation::Sub))
    }
    fn mul(&self, other: &Self) -> Result<Self, TensorOpError> {
        // Validate shape. matches_mul_size_op Guarantees that both left and right are matrices, meaning indexing [0] and [1] cannot panic.
        if !self.shape.matches_mul_size_op(&other.shape) {
            return Err(TensorOpError::shape_mismatch(TensorOperation::Mul, &self.shape, &other.shape));
        }

        // Shape is (our_batch, our height, other width)
        let (left_height, left_width) = {
            let size = self.shape.last_two_dims();
            (size[0], size[1])
        };
        let right_width = other.shape.last_two_dims()[1];

        let batch_dims = self.shape.batch_dims();
        let Some(output_shape) = TensorShape::from_batch_size(batch_dims, &[left_height, right_width]) else {
            return Err(TensorOpError::shape_mismatch(TensorOperation::Mul, &self.shape, &other.shape));
        };

        // matches_mul_size_op guarantees that left batch size is either equal to right batch size, OR
        // right does not have any batching.
        let output_matrix_size = left_height * right_width;
        let left_matrix_size = left_height * left_width;
        let right_matrix_size = left_width * right_width;
        let right_is_batched = !other.shape.batch_dims().is_empty();

        Ok(Self {
            data: (0..output_shape.num_elements()).map(|idx| {
                let (batch_idx, mat_idx) = (idx / output_matrix_size, idx % output_matrix_size);
                let (row, col) = (mat_idx / right_width, mat_idx % right_width);
                let left_batch_idx = batch_idx * left_matrix_size;
                let right_batch_idx = if right_is_batched { batch_idx * right_matrix_size } else { 0 };

                // Dot row in left with col in right.
                (0..left_width).map(|i| {
                    self.data[left_batch_idx + row * left_width + i] * other.data[right_batch_idx + i * right_width + col]
                }).sum::<f32>()
            }).collect(),
            shape: output_shape,
        })
    }
    fn mul_scalar(&self, other: f32) -> Result<Self, TensorOpError> {
        Ok(Self {
            data: self.data.iter().map(|v| v * other).collect(),
            shape: self.shape.clone(),
        })
    }

    fn to_cpu(&self) -> CPUTensor { self.clone() }
    fn shape(&self) -> &TensorShape { &self.shape }
}
