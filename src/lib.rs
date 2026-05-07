
// Chain rule:
// f(x) = a(b(x))
// f'(x) = a'(b(x)) * b'(x)

// Chain rule expanded:
// f(x) = a(b(c(d(e(f)))))
// f'(x) = a'(b(c(d(e(f))))) * b'(c(d(e(f)))) * c'(d(e(f))) * d'(e(f)) * e'(f)

mod shape;
pub use shape::TensorShape;

mod tensor;
pub use tensor::*;

mod gradient;
pub use gradient::*;
