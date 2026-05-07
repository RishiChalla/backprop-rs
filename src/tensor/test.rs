use super::{Tensor, TensorOutput};
use crate::{CPUTensor, TensorShape};

const EPSILON: f32 = 1e-5;

type TensorFactory<T> = fn(TensorShape, Vec<f32>) -> T;

fn tensor<T: Tensor>(factory: TensorFactory<T>, shape: &[usize], data: Vec<f32>) -> T {
    factory(TensorShape::from(shape.to_vec()), data)
}

fn cpu_tensor(shape: &[usize], data: Vec<f32>) -> CPUTensor {
    CPUTensor::from_data(TensorShape::from(shape.to_vec()), data).expect("test data should fill shape exactly")
}

fn expect_output<T: Tensor>(output: TensorOutput<T>) -> T {
    match output {
        TensorOutput::Tensor(tensor) => tensor,
        TensorOutput::Invalid(error) => panic!("expected tensor output, got error: {error}"),
    }
}

fn assert_tensor_eq<T: Tensor>(actual: T, expected: CPUTensor) {
    let actual = actual.to_cpu();
    assert!(
        actual.eq_epsilon(&expected, EPSILON),
        "tensor mismatch\nactual: {actual:?}\nexpected: {expected:?}"
    );
}

fn assert_output_eq<T: Tensor>(actual: TensorOutput<T>, expected: CPUTensor) {
    assert_tensor_eq(expect_output(actual), expected);
}

fn softmax_expected(data: &[f32]) -> Vec<f32> {
    let exp_sum = data.iter().copied().map(f32::exp).sum::<f32>();
    data.iter().map(|&x| f32::exp(x) / exp_sum).collect()
}

fn test_relu<T: Tensor>(factory: TensorFactory<T>) {
    assert_tensor_eq(
        tensor(factory, &[2, 3], vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]).relu(),
        cpu_tensor(&[2, 3], vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0]),
    );

    assert_tensor_eq(
        tensor(factory, &[2, 2, 3], vec![-1.0, 2.0, -3.0, 4.0, 0.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0]).relu(),
        cpu_tensor(&[2, 2, 3], vec![0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 6.0, 0.0, 8.0, 0.0, 10.0, 0.0]),
    );

    let wide_data = (-10..10).map(|x| x as f32).collect::<Vec<_>>();
    let wide_expected = wide_data.iter().map(|x| x.max(0.0)).collect::<Vec<_>>();
    assert_tensor_eq(
        tensor(factory, &[2, 10], wide_data).relu(),
        cpu_tensor(&[2, 10], wide_expected),
    );

    let tall_data = (-10..10).rev().map(|x| x as f32).collect::<Vec<_>>();
    let tall_expected = tall_data.iter().map(|x| x.max(0.0)).collect::<Vec<_>>();
    assert_tensor_eq(
        tensor(factory, &[10, 2], tall_data).relu(),
        cpu_tensor(&[10, 2], tall_expected),
    );
}

fn test_softmax<T: Tensor>(factory: TensorFactory<T>) {
    let data = vec![0.0, 1.0, 2.0];
    assert_tensor_eq(
        tensor(factory, &[3], data.clone()).softmax(),
        cpu_tensor(&[3], softmax_expected(&data)),
    );

    let batched_data = vec![0.0; 8];
    assert_tensor_eq(
        tensor(factory, &[2, 2, 2], batched_data.clone()).softmax(),
        cpu_tensor(&[2, 2, 2], softmax_expected(&batched_data)),
    );

    let wide_data = vec![0.0; 20];
    assert_tensor_eq(
        tensor(factory, &[2, 10], wide_data.clone()).softmax(),
        cpu_tensor(&[2, 10], softmax_expected(&wide_data)),
    );

    let tall_data = vec![0.0; 20];
    assert_tensor_eq(
        tensor(factory, &[10, 2], tall_data.clone()).softmax(),
        cpu_tensor(&[10, 2], softmax_expected(&tall_data)),
    );
}

fn test_neg<T: Tensor>(factory: TensorFactory<T>) {
    assert_tensor_eq(
        tensor(factory, &[2, 3], vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]).neg(),
        cpu_tensor(&[2, 3], vec![2.0, 1.0, -0.0, -1.0, -2.0, -3.0]),
    );

    assert_tensor_eq(
        tensor(factory, &[2, 2, 3], vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0]).neg(),
        cpu_tensor(&[2, 2, 3], vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0]),
    );

    assert_tensor_eq(
        tensor(factory, &[2, 10], vec![1.0; 20]).neg(),
        cpu_tensor(&[2, 10], vec![-1.0; 20]),
    );

    assert_tensor_eq(
        tensor(factory, &[10, 2], vec![-1.0; 20]).neg(),
        cpu_tensor(&[10, 2], vec![1.0; 20]),
    );
}

fn test_add<T: Tensor>(factory: TensorFactory<T>) {
    assert_output_eq(
        tensor(factory, &[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .add(&tensor(factory, &[2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])),
        cpu_tensor(&[2, 3], vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]),
    );

    assert_output_eq(
        tensor(factory, &[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .add(&tensor(factory, &[2, 2], vec![10.0, 20.0, 30.0, 40.0])),
        cpu_tensor(&[2, 2, 2], vec![11.0, 22.0, 33.0, 44.0, 15.0, 26.0, 37.0, 48.0]),
    );

    assert_output_eq(
        tensor(factory, &[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .add(&tensor(factory, &[2, 2, 2], vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])),
        cpu_tensor(&[2, 2, 2], vec![9.0; 8]),
    );

    assert_output_eq(
        tensor(factory, &[2, 10], vec![1.0; 20]).add(&tensor(factory, &[2, 10], vec![2.0; 20])),
        cpu_tensor(&[2, 10], vec![3.0; 20]),
    );

    assert_output_eq(
        tensor(factory, &[10, 2], vec![1.0; 20]).add(&tensor(factory, &[10, 2], vec![2.0; 20])),
        cpu_tensor(&[10, 2], vec![3.0; 20]),
    );
}

fn test_sub<T: Tensor>(factory: TensorFactory<T>) {
    assert_output_eq(
        tensor(factory, &[2, 3], vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0])
            .sub(&tensor(factory, &[2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])),
        cpu_tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    );

    assert_output_eq(
        tensor(factory, &[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .sub(&tensor(factory, &[2, 2], vec![10.0, 20.0, 30.0, 40.0])),
        cpu_tensor(&[2, 2, 2], vec![-9.0, -18.0, -27.0, -36.0, -5.0, -14.0, -23.0, -32.0]),
    );

    assert_output_eq(
        tensor(factory, &[2, 2, 2], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
            .sub(&tensor(factory, &[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])),
        cpu_tensor(&[2, 2, 2], vec![9.0, 18.0, 27.0, 36.0, 45.0, 54.0, 63.0, 72.0]),
    );

    assert_output_eq(
        tensor(factory, &[2, 10], vec![3.0; 20]).sub(&tensor(factory, &[2, 10], vec![1.0; 20])),
        cpu_tensor(&[2, 10], vec![2.0; 20]),
    );

    assert_output_eq(
        tensor(factory, &[10, 2], vec![3.0; 20]).sub(&tensor(factory, &[10, 2], vec![1.0; 20])),
        cpu_tensor(&[10, 2], vec![2.0; 20]),
    );
}

fn test_mul<T: Tensor>(factory: TensorFactory<T>) {
    assert_output_eq(
        tensor(factory, &[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .mul(&tensor(factory, &[3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])),
        cpu_tensor(&[2, 2], vec![58.0, 64.0, 139.0, 154.0]),
    );

    assert_output_eq(
        tensor(factory, &[2, 2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            .mul(&tensor(factory, &[3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])),
        cpu_tensor(&[2, 2, 2], vec![58.0, 64.0, 139.0, 154.0, 220.0, 244.0, 301.0, 334.0]),
    );

    assert_output_eq(
        tensor(factory, &[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .mul(&tensor(factory, &[2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
        cpu_tensor(&[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 11.0, 11.0, 15.0, 15.0]),
    );

    assert_output_eq(
        tensor(factory, &[2, 10], vec![1.0; 20]).mul(&tensor(factory, &[10, 2], vec![1.0; 20])),
        cpu_tensor(&[2, 2], vec![10.0; 4]),
    );

    assert_output_eq(
        tensor(factory, &[10, 2], vec![1.0; 20]).mul(&tensor(factory, &[2, 10], vec![1.0; 20])),
        cpu_tensor(&[10, 10], vec![2.0; 100]),
    );

    assert_output_eq(
        tensor(factory, &[2, 2, 3], vec![1.0; 12]).mul(&tensor(factory, &[3, 4], vec![1.0; 12])),
        cpu_tensor(&[2, 2, 4], vec![3.0; 16]),
    );

    assert_output_eq(
        tensor(factory, &[2, 2, 3], vec![1.0; 12]).mul(&tensor(factory, &[2, 3, 4], [vec![1.0; 12], vec![2.0; 12]].concat())),
        cpu_tensor(&[2, 2, 4], [vec![3.0; 8], vec![6.0; 8]].concat()),
    );
}

#[test]
fn test_cpu() {
    fn make_cpu(shape: TensorShape, data: Vec<f32>) -> CPUTensor {
        CPUTensor::from_data(shape, data).expect("test data should fill CPU tensor shape exactly")
    }

    test_relu::<CPUTensor>(make_cpu);
    test_softmax::<CPUTensor>(make_cpu);
    test_neg::<CPUTensor>(make_cpu);
    test_add::<CPUTensor>(make_cpu);
    test_sub::<CPUTensor>(make_cpu);
    test_mul::<CPUTensor>(make_cpu);
}
