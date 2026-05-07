
// Chain rule:
// f(x) = a(b(x))
// f'(x) = a'(b(x)) * b'(x)

// Chain rule expanded:
// f(x) = a(b(c(d(e(f)))))
// f'(x) = a'(b(c(d(e(f))))) * b'(c(d(e(f)))) * c'(d(e(f))) * d'(e(f)) * e'(f)

use crate::{CPUTensor, ConstructableTensor, Tensor, TensorOutput, TensorShape};

#[derive(Debug, PartialEq, Clone)]
pub struct WithGradient<T: Tensor> {
    pub(crate) tensor: T,
    pub(crate) gradient: T,
}

impl<T: ConstructableTensor> From<T> for WithGradient<T> {
    fn from(tensor: T) -> Self {
        Self {
            // Derivative of f(x) = x is 1. This is the default intialized gradient for the input tensor,
            // all further operations on this tensor will have gradients tracked.
            gradient: T::from_scalar(tensor.shape().clone(), 1.0),
            tensor,
        }
    }
}

// impl<T: Tensor> Tensor for WithGradient<T> {
//     fn relu(&self) -> Self {
//         Self {
//             tensor: self.tensor.relu(),
//             gradient: self.tensor.relu_d() * self.gradient,
//         }
//     }

//     fn softmax(&self) -> Self {
//         todo!()
//     }

//     fn neg(&self) -> Self {
//         todo!()
//     }

//     fn add(&self, other: &Self) -> TensorOutput<Self> {
//         todo!()
//     }

//     fn sub(&self, other: &Self) -> TensorOutput<Self> {
//         todo!()
//     }

//     fn mul(&self, other: &Self) -> TensorOutput<Self> {
//         todo!()
//     }

//     fn to_cpu(&self) -> CPUTensor {
//         todo!()
//     }

//     fn shape(&self) -> &TensorShape {
//         todo!()
//     }
// }
