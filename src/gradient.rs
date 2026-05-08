
// Chain rule:
// f(x) = a(b(x))
// f'(x) = a'(b(x)) * b'(x)

// Chain rule expanded:
// f(x) = a(b(c(d(e(f)))))
// f'(x) = a'(b(c(d(e(f))))) * b'(c(d(e(f)))) * c'(d(e(f))) * d'(e(f)) * e'(f)

use crate::{CPUTensor, ConstructableTensor, Tensor, TensorOpError, TensorOperation, TensorShape};

#[derive(Debug, PartialEq, Clone)]
pub struct WithGradient<T: ConstructableTensor> {
    pub(crate) tensor: T,
    pub(crate) gradient: T,
}

impl<T: ConstructableTensor> WithGradient<T> {
    fn with_zero_grad(&self, tensor: T) -> Result<Self, TensorOpError> {
        Ok(Self { tensor, gradient: T::from_scalar(self.gradient.shape().clone(), 0.0) })
    }
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

impl<T: ConstructableTensor> Tensor for WithGradient<T> {
    // Single-var ops
    fn relu(&self) -> Result<Self, TensorOpError> {
        Ok(Self {
            tensor: self.tensor.relu()?,
            // f(x) = relu(a(x))
            // f'(x) = f32(a(x) > 0) * a'(x)
            gradient: self.tensor.gt_scalar(0.0).map_err(|err| err.with_op(TensorOperation::Relu))?
                .mul(&self.gradient).map_err(|err| err.with_op(TensorOperation::Relu))?,
        })
    }
    fn neg(&self) -> Result<Self, TensorOpError> {
        Ok(Self {
            tensor: self.tensor.neg()?,
            // f(x) = -1 * a(x)
            // f'(x) = -1 * a'(x)
            gradient: self.gradient.neg()?,
        })
    }
    fn abs(&self) -> Result<Self, TensorOpError> {
        Ok(Self {
            tensor: self.tensor.abs()?,
            // f(x) = abs(a(x))
            // f'(x) = sign(x) * a'(x)
            gradient: self.tensor.sign().map_err(|err| err.with_op(TensorOperation::Abs))?
                .mul(&self.gradient).map_err(|err| err.with_op(TensorOperation::Abs))?,
        })
    }
    fn exp(&self) -> Result<Self, TensorOpError> {
        Ok(Self {
            tensor: self.tensor.exp()?,
            // f(x) = exp(a(x))
            // f'(x) = exp(a(x)) * a'(x)
            gradient: self.tensor.exp()?
                .mul(&self.gradient).map_err(|err| err.with_op(TensorOperation::Exp))?,
        })
    }
    fn sign(&self) -> Result<Self, TensorOpError> { self.with_zero_grad(self.tensor.sign()?) }

    // Scalar Boolean operations
    fn gt_scalar(&self, scalar: f32) -> Result<Self, TensorOpError> { self.with_zero_grad(self.tensor.gt_scalar(scalar)?) }
    fn gte_scalar(&self, scalar: f32) -> Result<Self, TensorOpError> { self.with_zero_grad(self.tensor.gte_scalar(scalar)?) }
    fn lt_scalar(&self, scalar: f32) -> Result<Self, TensorOpError> { self.with_zero_grad(self.tensor.lt_scalar(scalar)?) }
    fn lte_scalar(&self, scalar: f32) -> Result<Self, TensorOpError> { self.with_zero_grad(self.tensor.lte_scalar(scalar)?) }
    fn eq_scalar(&self, scalar: f32) -> Result<Self, TensorOpError> { self.with_zero_grad(self.tensor.eq_scalar(scalar)?) }

    // Multi-var Boolean operations
    fn gt(&self, other: &Self) -> Result<Self, TensorOpError> { self.with_zero_grad(self.tensor.gt(&other.tensor)?) }
    fn gte(&self, other: &Self) -> Result<Self, TensorOpError> { self.with_zero_grad(self.tensor.gte(&other.tensor)?) }
    fn lt(&self, other: &Self) -> Result<Self, TensorOpError> { self.with_zero_grad(self.tensor.lt(&other.tensor)?) }
    fn lte(&self, other: &Self) -> Result<Self, TensorOpError> { self.with_zero_grad(self.tensor.lte(&other.tensor)?) }
    fn eq(&self, other: &Self) -> Result<Self, TensorOpError> { self.with_zero_grad(self.tensor.eq(&other.tensor)?) }

    // Multi-var ops
    fn add(&self, other: &Self) -> Result<Self, TensorOpError> {
        Ok(Self {
            tensor: self.tensor.add(&other.tensor)?,
            // f(x) = a(x) + b(x)
            // f'(x) = a'(x) + b'(x)
            gradient: self.gradient.add(&other.gradient)?,
        })
    }
    fn sub(&self, other: &Self) -> Result<Self, TensorOpError> {
        Ok(Self {
            tensor: self.tensor.sub(&other.tensor)?,
            // f(x) = a(x) - b(x)
            // f'(x) = a'(x) - b'(x)
            gradient: self.gradient.sub(&other.gradient)?,
        })
    }
    fn mul(&self, other: &Self) -> Result<Self, TensorOpError> {
        Ok(Self {
            tensor: self.tensor.mul(&other.tensor)?,
            // f(x) = a(x)b(x)
            // f'(x) = a'(x)b(x) + a(x)b'(x)
            gradient: self.gradient.mul(&other.tensor)?
                .add(&self.tensor.mul(&other.gradient)?)
                .map_err(|err| err.with_op(TensorOperation::Mul))?,
        })
    }
    fn mul_scalar(&self, scalar: f32) -> Result<Self, TensorOpError> {
        Ok(Self {
            tensor: self.tensor.mul_scalar(scalar)?,
            // f(x) = N * a(x)
            // f'(x) = N * a'(x)
            gradient: self.gradient.mul_scalar(scalar)?,
        })
    }
    fn cross_entropy_loss_logits(&self, target: &Self) -> Result<Self, TensorOpError> {
        todo!()
    }

    /// Detaches the Tensor from further gradient calculation and returns it on the CPU.
    fn to_cpu(&self) -> CPUTensor { self.tensor.to_cpu() }
    fn shape(&self) -> &TensorShape { self.tensor.shape() }
}
