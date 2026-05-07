//! This module provides a clean no-panic interface for tensor operations,
//! with control flow and conditional chaining.

use std::{fmt::Display, ops::{Add, AddAssign, Mul, Sub, SubAssign}};

use thiserror::Error;

use crate::{Tensor, TensorShape};

// >----------------------------------------------- Operations & Errors -----------------------------------------------<

/// An operation performed on tensors
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(crate) enum TensorOperation {
    Relu,
    Softmax,
    Neg,
    Add,
    Mul,
    Sub,
    Div,
}

/// An error that occurs during a tensor operation.
#[derive(Debug, Clone)]
pub struct TensorOpError {
    pub(crate) operation: TensorOperation,
    pub(crate) error: TensorOpErrorVariant,
}

impl TensorOpError {
    pub(crate) fn with_op(mut self, op: TensorOperation) -> Self {
        self.operation = op;
        self
    }

    pub(crate) fn shape_mismatch(op: TensorOperation, shape: &TensorShape, other_shape: &TensorShape) -> Self {
        Self {
            operation: op,
            error: TensorOpErrorVariant::ShapeMismatch { shape: shape.clone(), other_shape: other_shape.clone() },
        }
    }
}

impl Display for TensorOpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error during {:?} operation - {}", self.operation, self.error)
    }
}

/// Error variants for issues that occur during a tensor operation
#[derive(Debug, Error, Clone)]
pub enum TensorOpErrorVariant {
    #[error("Invalid shape: Our shape={shape} - Other's shape={other_shape}")]
    ShapeMismatch { shape: TensorShape, other_shape: TensorShape },
}

// >----------------------------------------------- Tensor Output -----------------------------------------------<

/// The output of a tensor operation. May contain an error. We use a custom enum instead of Result to support
/// Conditional chaining for operator overloading.
pub enum TensorOutput<T: Tensor> {
    Tensor(T),
    Invalid(TensorOpError),
}

impl<T: Tensor> TensorOutput<T> {
    pub fn ok(self) -> Result<T, TensorOpError> {
        match self {
            Self::Tensor(t) => Ok(t),
            Self::Invalid(e) => Err(e),
        }
    }

    /// Performs an operation on this and another (rhs) tensor given a closure to perform the operation.
    pub fn op(&self, other: &Self, op: impl Fn(&T, &T) -> Result<T, TensorOpError>) -> Self {
        match (self, other) {
            // Both left and right are valid - perform operation
            (Self::Tensor(left), Self::Tensor(right)) => op(left, right).into(),
            // One of the sides were invalid - propogate the error
            (Self::Tensor(_), Self::Invalid(error))
             | (Self::Invalid(error), Self::Tensor(_))
             | (Self::Invalid(error), Self::Invalid(_)) => Self::Invalid(error.clone()),
        }
    }
}

impl<T: Tensor> From<TensorOutput<T>> for Result<T, TensorOpError> {
    fn from(value: TensorOutput<T>) -> Self { value.ok() }
}

impl<T: Tensor> From<Result<T, TensorOpError>> for TensorOutput<T> {
    fn from(value: Result<T, TensorOpError>) -> Self {
        match value {
            Ok(t) => Self::Tensor(t),
            Err(e) => Self::Invalid(e),
        }
    }
}

// >----------------------------------------------- Operator Overloading -----------------------------------------------<

impl<T: Tensor> Add<TensorOutput<T>> for TensorOutput<T> {
    type Output = Self;
    fn add(self, rhs: TensorOutput<T>) -> Self { self.op(&rhs, |l, r| l.add(r)) }
}

impl<T: Tensor> Sub<TensorOutput<T>> for TensorOutput<T> {
    type Output = Self;
    fn sub(self, rhs: TensorOutput<T>) -> Self { self.op(&rhs, |l, r| l.sub(r)) }
}

impl<T: Tensor> Mul<TensorOutput<T>> for TensorOutput<T> {
    type Output = Self;
    fn mul(self, rhs: TensorOutput<T>) -> Self { self.op(&rhs, |l, r| l.mul(r)) }
}

// Note / Out of scope - Assign operators are technically a bit inefficient since they make a temporary copy
// instead of mutating self. Should not be a major performance concern as this only moderately linearly increases
// performance cost, and production-grade is not a crate goal.

impl<T: Tensor> AddAssign<TensorOutput<T>> for TensorOutput<T> {
    fn add_assign(&mut self, rhs: TensorOutput<T>) { *self = self.op(&rhs, |l, r| l.add(r)); }
}

impl<T: Tensor> SubAssign<TensorOutput<T>> for TensorOutput<T> {
    fn sub_assign(&mut self, rhs: TensorOutput<T>) { *self = self.op(&rhs, |l, r| l.sub(r)); }
}
