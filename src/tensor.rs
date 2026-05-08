
use crate::{TensorShape, WithGradient};

mod cpu;
pub use cpu::*;

mod ops;
pub use ops::*;

#[cfg(test)]
mod test;

/// Represents a tensor, a multi-dimensional array of values.
/// Backend may be implemented by different types, such as [`CPUTensor`].
/// All operations are per-batch.
pub trait Tensor: Sized {
    // Single-var operations
    fn relu(&self) -> Result<Self, TensorOpError>;
    fn neg(&self) -> Result<Self, TensorOpError>;
    fn abs(&self) -> Result<Self, TensorOpError>;
    fn exp(&self) -> Result<Self, TensorOpError>;
    /// -1 for all elements < 0, 1 for all elements > 0. Uses sign bit for 0
    fn sign(&self) -> Result<Self, TensorOpError>;

    // Scalar Boolean operations
    /// Returns 1.0 for all elements greater than the scalar. 0.0 For all elements less than or equal to.
    fn gt_scalar(&self, scalar: f32) -> Result<Self, TensorOpError>;
    /// Returns 1.0 for all elements greater than or equal to the scalar. 0.0 For all elements less than.
    fn gte_scalar(&self, scalar: f32) -> Result<Self, TensorOpError>;
    /// Returns 1.0 for all elements less than the scalar. 0.0 For all elements greater than or equal to.
    fn lt_scalar(&self, scalar: f32) -> Result<Self, TensorOpError>;
    /// Returns 1.0 for all elements less than or equal to the scalar. 0.0 For all elements greater than.
    fn lte_scalar(&self, scalar: f32) -> Result<Self, TensorOpError>;
    /// Returns 1.0 for all elements equal to the corresponding tensor value. 0.0 For all elements unequal.
    /// This does not have any epsilon logic, if you need a degree of error you must implement it yourself.
    fn eq_scalar(&self, scalar: f32) -> Result<Self, TensorOpError>;

    // Multi-var Boolean operations
    /// Returns 1.0 for all elements greater than the corresponding tensor value. 0.0 For all elements less than or equal to.
    fn gt(&self, other: &Self) -> Result<Self, TensorOpError>;
    /// Returns 1.0 for all elements greater than or equal to the corresponding tensor value. 0.0 For all elements less than.
    fn gte(&self, other: &Self) -> Result<Self, TensorOpError>;
    /// Returns 1.0 for all elements less than the corresponding tensor value. 0.0 For all elements greater than or equal to.
    fn lt(&self, other: &Self) -> Result<Self, TensorOpError>;
    /// Returns 1.0 for all elements less than or equal to the corresponding tensor value. 0.0 For all elements greater than.
    fn lte(&self, other: &Self) -> Result<Self, TensorOpError>;
    /// Returns 1.0 for all elements equal to the corresponding tensor value. 0.0 For all elements unequal.
    /// This does not have any epsilon logic, if you need a degree of error you must implement it yourself.
    fn eq(&self, other: &Self) -> Result<Self, TensorOpError>;

    // Multi-var operations
    fn add(&self, other: &Self) -> Result<Self, TensorOpError>;
    fn sub(&self, other: &Self) -> Result<Self, TensorOpError>;
    fn mul(&self, other: &Self) -> Result<Self, TensorOpError>;
    fn mul_scalar(&self, scalar: f32) -> Result<Self, TensorOpError>;
    /// Assumes the current tensor (left hand side) is a logits tensor, and performs softmax (per batch).
    /// Then calculates the cross entropy loss towards the given target probability distribution.
    /// This is averaged across all batches, resulting in the output of this operation being shape [1]
    fn cross_entropy_loss_logits(&self, target: &Self) -> Result<Self, TensorOpError>;

    // Converts the tensor to an easily CPU readable tensor, useful for extracting data
    fn to_cpu(&self) -> CPUTensor;

    fn shape(&self) -> &TensorShape;
    fn to_output(self) -> Result<Self, TensorOpError> { Ok(self) }

    /// Starts tracking gradients for further operations on this tensor.
    fn with_grad(self) -> WithGradient<Self> where Self: ConstructableTensor { self.into() }
}

/// Tensors that support creation. Some tensor types (ie Gradients) must be converted from CPU tensor types,
/// and cannot be statically initialized.
pub trait ConstructableTensor: Tensor {
    /// Creates a zero-initialized tensor of the given shape.
    fn new(shape: TensorShape) -> Self;
    /// Creates an initialized tensor of the given shape filled with the given scalar.
    fn from_scalar(shape: TensorShape, scalar: f32) -> Self;
}
