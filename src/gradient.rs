use crate::Tensor;

pub struct WithGradient<T: Tensor> {
    tensor: T,
    gradient: T,
}

impl<T: Tensor> WithGradient<T> {
    fn new(tensor: T) -> Self {
        Self {
            // Derivative of f(x) = x is 1. This is the default intialized gradient for the input tensor,
            // all further operations on this tensor will have gradients tracked.
            gradient: T::from_scalar(tensor.shape().clone(), 1.0),
            tensor,
        }
    }
}
