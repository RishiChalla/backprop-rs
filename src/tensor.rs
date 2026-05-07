
use crate::{TensorShape, WithGradient};

mod cpu;
pub use cpu::*;

mod ops;
pub use ops::*;

#[cfg(test)]
mod test;

/// Represents a tensor, a multi-dimensional array of values.
/// Backend may be implemented by different types, such as [`CPUTensor`].
pub trait Tensor: PartialEq + Sized {
    // Single-var operations
    fn relu(&self) -> TensorOutput<Self>;
    fn softmax(&self) -> TensorOutput<Self>;
    fn neg(&self) -> TensorOutput<Self>;

    // Multi-var operations
    fn add(&self, other: &Self) -> TensorOutput<Self>;
    fn sub(&self, other: &Self) -> TensorOutput<Self>;
    fn mul(&self, other: &Self) -> TensorOutput<Self>;
    fn mul_scalar(&self, other: f32) -> TensorOutput<Self>;

    // Converts the tensor to an easily CPU readable tensor, useful for extracting data
    fn to_cpu(&self) -> CPUTensor;

    fn shape(&self) -> &TensorShape;
    fn to_output(self) -> TensorOutput<Self> { TensorOutput::Tensor(self) }

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

/// Tensors that can have their gradients calculated. Every operation supports gradient calculation.
/// Note - this trait is required to be implemented on any tensor that works with [`WithGradient`],
/// but just implementing this trait alone does not mean your tensor will automatically calculate gradients.
/// You must use WithGradient or equivelant to calculate gradients.
pub trait DifferentiableTensor: Tensor {
    // Single-var operations
    fn relu_d(&self) -> Self;
    fn softmax_d(&self) -> Self;
    fn neg_d(&self) -> Self;

    // Multi-var operations
    fn add_d(&self, other: &Self) -> TensorOutput<Self>;
    fn sub_d(&self, other: &Self) -> TensorOutput<Self>;
    fn mul_d(&self, other: &Self) -> TensorOutput<Self>;
    fn mul_scalar_d(&self, other: f32) -> TensorOutput<Self>;
}
