use crate::{Tensor, TensorOutput, TensorShape, tensor::TensorOperation};


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

impl CPUTensor {
    /// Creates a zero-initialized tensor of the given shape.
    pub fn new(shape: TensorShape) -> Self {
        Self {
            data: vec![0.0; shape.num_elements()],
            shape,
        }
    }
}

impl Tensor for CPUTensor {
    fn shape(&self) -> &TensorShape { &self.shape }

    // Single-var operations
    fn relu(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| x.max(0.0)).collect(),
        }
    }
    fn softmax(&self) -> Self {
        let exp_sum = self.data.iter().cloned().map(f32::exp).sum::<f32>();
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| f32::exp(x) / exp_sum).collect(),
        }
    }
    fn neg(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| -x).collect(),
        }
    }

    // Multi-var operations
    fn add(&self, other: &Self) -> TensorOutput<Self> {
        // Validate that shape matches
        if self.shape.matches_batch_size_op(&other.shape) {
            return TensorOutput::shape_mismatch(TensorOperation::Add, &self.shape, &other.shape);
        }

        TensorOutput::Tensor(Self {
            shape: self.shape.clone(),
            data: self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect(),
        })
    }
    fn sub(&self, other: &Self) -> TensorOutput<Self> {
        let mut result = self.add(&other.neg());
        // Update operation in error
        if let TensorOutput::Invalid(error) = &mut result { error.operation = TensorOperation::Sub; }
        result
    }
    fn mul(&self, other: &Self) -> TensorOutput<Self> {
        // Validate shape
        if !self.shape.matches_mul_size_op(&other.shape) {
            return TensorOutput::shape_mismatch(TensorOperation::Mul, &self.shape, &other.shape);
        }

        // Perform matrix multiplication
        todo!()
    }
}
