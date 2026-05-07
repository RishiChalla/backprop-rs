
// Chain rule:
// f(x) = a(b(x))
// f'(x) = a'(b(x)) * b'(x)

// Chain rule expanded:
// f(x) = a(b(c(d(e(f)))))
// f'(x) = a'(b(c(d(e(f))))) * b'(c(d(e(f)))) * c'(d(e(f))) * d'(e(f)) * e'(f)

use crate::{CPUTensor, ConstructableTensor, DifferentiableTensor, Tensor, TensorOutput, TensorShape};

#[derive(Debug, PartialEq, Clone)]
pub struct WithGradient<T: ConstructableTensor + DifferentiableTensor> {
    pub(crate) tensor: T,
    pub(crate) gradient: T,
}

impl<T: ConstructableTensor + DifferentiableTensor> From<T> for WithGradient<T> {
    fn from(tensor: T) -> Self {
        Self {
            // Derivative of f(x) = x is 1. This is the default intialized gradient for the input tensor,
            // all further operations on this tensor will have gradients tracked.
            gradient: T::from_scalar(tensor.shape().clone(), 1.0),
            tensor,
        }
    }
}
