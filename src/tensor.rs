
use std::fmt::Display;

use thiserror::Error;

use crate::TensorShape;

mod cpu;
pub use cpu::*;

#[cfg(test)]
mod test;

/// Represents a tensor, a multi-dimensional array of values.
/// Backend may be implemented by different types, such as [`CPUTensor`].
pub trait Tensor: PartialEq + Sized {
    fn shape(&self) -> &TensorShape;

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
}

/// The output of a tensor operation. May contain an error. We use a custom enum instead of Result to support
/// Conditional chaining for operator overloading.
pub enum TensorOutput<T: Tensor> {
    Tensor(T),
    Invalid(TensorOpError),
}

impl<T: Tensor> TensorOutput<T> {
    /// Creates a new Shape mismatch error
    fn shape_mismatch(op: TensorOperation, shape: &TensorShape, other_shape: &TensorShape) -> Self {
        TensorOutput::Invalid(TensorOpError {
            operation: op,
            error: TensorOpErrorVariant::ShapeMismatch {
                shape: shape.clone(),
                other_shape: other_shape.clone(),
            },
        })
    }
}

/// An error that occurs during a tensor operation.
#[derive(Debug)]
pub struct TensorOpError {
    operation: TensorOperation,
    error: TensorOpErrorVariant,
}

/// An operation performed on tensors
#[derive(Debug)]
#[allow(dead_code)]
enum TensorOperation {
    Relu,
    Softmax,
    Neg,
    Add,
    Mul,
    Sub,
    Div,
}

impl Display for TensorOpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error during {:?} operation - {}", self.operation, self.error)
    }
}

/// Error variants for issues that occur during a tensor operation
#[derive(Debug, Error)]
pub enum TensorOpErrorVariant {
    #[error("Invalid shape: Our shape={shape} - Other's shape={other_shape}")]
    ShapeMismatch { shape: TensorShape, other_shape: TensorShape },
}
