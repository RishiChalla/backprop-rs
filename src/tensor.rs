
use crate::TensorShape;

mod cpu;
pub use cpu::*;

mod ops;
pub use ops::*;

#[cfg(test)]
mod test;

/// Represents a tensor, a multi-dimensional array of values.
/// Backend may be implemented by different types, such as [`CPUTensor`].
pub trait Tensor: PartialEq + Sized {
    // Constructors
    fn new(shape: TensorShape) -> Self;
    fn from_scalar(shape: TensorShape, scalar: f32) -> Self;

    // Single-var operations
    fn relu(&self) -> Self;
    fn softmax(&self) -> Self;
    fn neg(&self) -> Self;

    // Multi-var operations
    fn add(&self, other: &Self) -> TensorOutput<Self>;
    fn sub(&self, other: &Self) -> TensorOutput<Self>;
    fn mul(&self, other: &Self) -> TensorOutput<Self>;

    // Converts the tensor to an easily CPU readable tensor, useful for extracting data
    fn to_cpu(&self) -> CPUTensor;

    fn shape(&self) -> &TensorShape;
    fn to_output(self) -> TensorOutput<Self> { TensorOutput::Tensor(self) }
}
