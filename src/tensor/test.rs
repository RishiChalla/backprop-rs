
use crate::{CPUTensor, Tensor, TensorOpError, TensorShape};

const EPSILON: f32 = 1e-5;

type TensorFactory<T> = fn(TensorShape, Vec<f32>) -> T;

fn tensor<T: Tensor>(factory: TensorFactory<T>, shape: &[usize], data: Vec<f32>) -> T {
    factory(TensorShape::from(shape.to_vec()), data)
}

fn cpu_tensor(shape: &[usize], data: Vec<f32>) -> CPUTensor {
    CPUTensor::from_data(TensorShape::from(shape.to_vec()), data).expect("test data should fill shape exactly")
}

fn expect_output<T: Tensor>(output: Result<T, TensorOpError>) -> T {
    match output {
        Ok(tensor) => tensor,
        Err(error) => panic!("expected tensor output, got error: {error}"),
    }
}

fn assert_tensor_eq<T: Tensor>(actual: T, expected: CPUTensor) {
    let actual = actual.to_cpu();
    assert!(
        actual.eq_epsilon(&expected, EPSILON),
        "tensor mismatch\nactual: {actual:?}\nexpected: {expected:?}"
    );
}

fn assert_output_eq<T: Tensor>(actual: Result<T, TensorOpError>, expected: CPUTensor) {
    assert_tensor_eq(expect_output(actual), expected);
}

fn test_relu<T: Tensor>(factory: TensorFactory<T>) {
    // No batching: ReLU should preserve positive matrix entries and clamp negative ones.
    assert_output_eq(
        tensor(factory, &[2, 3], vec![
            -2.0, -1.0, 0.0,
             1.0,  2.0, 3.0,
        ]).relu(),
        cpu_tensor(&[2, 3], vec![
            0.0, 0.0, 0.0,
            1.0, 2.0, 3.0,
        ]),
    );

    // Batched input: element-wise ops should keep batch boundaries irrelevant to the value rule.
    assert_output_eq(
        tensor(factory, &[2, 2, 3], vec![
            // batch 0
            -1.0,  2.0, -3.0,
             4.0,  0.0, -5.0,
            // batch 1
             6.0, -7.0,  8.0,
            -9.0, 10.0, -11.0,
        ]).relu(),
        cpu_tensor(&[2, 2, 3], vec![
            // batch 0
            0.0, 2.0, 0.0,
            4.0, 0.0, 0.0,
            // batch 1
            6.0,  0.0, 8.0,
            0.0, 10.0, 0.0,
        ]),
    );

    // Wide matrix: catches index assumptions that only work for square-ish shapes.
    assert_output_eq(
        tensor(factory, &[2, 10], vec![
            -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
              0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
        ]).relu(),
        cpu_tensor(&[2, 10], vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        ]),
    );

    // Tall matrix: pairs with the wide case so row/column indexing is tested both ways.
    assert_output_eq(
        tensor(factory, &[10, 2], vec![
             9.0,  8.0,
             7.0,  6.0,
             5.0,  4.0,
             3.0,  2.0,
             1.0,  0.0,
            -1.0, -2.0,
            -3.0, -4.0,
            -5.0, -6.0,
            -7.0, -8.0,
            -9.0, -10.0,
        ]).relu(),
        cpu_tensor(&[10, 2], vec![
            9.0, 8.0,
            7.0, 6.0,
            5.0, 4.0,
            3.0, 2.0,
            1.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
        ]),
    );
}

fn test_neg<T: Tensor>(factory: TensorFactory<T>) {
    // No batching: negation should flip signs without changing shape.
    assert_output_eq(
        tensor(factory, &[2, 3], vec![
            -2.0, -1.0, 0.0,
             1.0,  2.0, 3.0,
        ]).neg(),
        cpu_tensor(&[2, 3], vec![
             2.0,  1.0, -0.0,
            -1.0, -2.0, -3.0,
        ]),
    );

    // Batched input: sign changes should not depend on batch position.
    assert_output_eq(
        tensor(factory, &[2, 2, 3], vec![
            // batch 0
             1.0, -2.0,  3.0,
            -4.0,  5.0, -6.0,
            // batch 1
             7.0,  -8.0,   9.0,
            -10.0, 11.0, -12.0,
        ]).neg(),
        cpu_tensor(&[2, 2, 3], vec![
            // batch 0
            -1.0,  2.0, -3.0,
             4.0, -5.0,  6.0,
            // batch 1
            -7.0,   8.0,  -9.0,
            10.0, -11.0,  12.0,
        ]),
    );

    // Wide matrix: validates sign flipping across long contiguous rows.
    assert_output_eq(
        tensor(factory, &[2, 10], vec![1.0; 20]).neg(),
        cpu_tensor(&[2, 10], vec![-1.0; 20]),
    );

    // Tall matrix: validates sign flipping across many short rows.
    assert_output_eq(
        tensor(factory, &[10, 2], vec![-1.0; 20]).neg(),
        cpu_tensor(&[10, 2], vec![1.0; 20]),
    );
}

fn test_add<T: Tensor>(factory: TensorFactory<T>) {
    // No batching: same-shape matrix addition should be element-wise.
    assert_output_eq(
        tensor(factory, &[2, 3], vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]).add(&tensor(factory, &[2, 3], vec![
            10.0, 20.0, 30.0,
            40.0, 50.0, 60.0,
        ])),
        cpu_tensor(&[2, 3], vec![
            11.0, 22.0, 33.0,
            44.0, 55.0, 66.0,
        ]),
    );

    // Left-only batching: the unbatched right operand should be reused for each left batch.
    assert_output_eq(
        tensor(factory, &[2, 2, 2], vec![
            // batch 0
            1.0, 2.0,
            3.0, 4.0,
            // batch 1
            5.0, 6.0,
            7.0, 8.0,
        ]).add(&tensor(factory, &[2, 2], vec![
            10.0, 20.0,
            30.0, 40.0,
        ])),
        cpu_tensor(&[2, 2, 2], vec![
            // batch 0
            11.0, 22.0,
            33.0, 44.0,
            // batch 1
            15.0, 26.0,
            37.0, 48.0,
        ]),
    );

    // Matching batch sizes: each left batch should pair with the corresponding right batch.
    assert_output_eq(
        tensor(factory, &[2, 2, 2], vec![
            // batch 0
            1.0, 2.0,
            3.0, 4.0,
            // batch 1
            5.0, 6.0,
            7.0, 8.0,
        ]).add(&tensor(factory, &[2, 2, 2], vec![
            // batch 0
            8.0, 7.0,
            6.0, 5.0,
            // batch 1
            4.0, 3.0,
            2.0, 1.0,
        ])),
        cpu_tensor(&[2, 2, 2], vec![9.0; 8]),
    );

    // Wide matrix: long rows catch row-major indexing mistakes in element-wise ops.
    assert_output_eq(
        tensor(factory, &[2, 10], vec![1.0; 20]).add(&tensor(factory, &[2, 10], vec![2.0; 20])),
        cpu_tensor(&[2, 10], vec![3.0; 20]),
    );

    // Tall matrix: many short rows pair with the wide test to check the opposite aspect ratio.
    assert_output_eq(
        tensor(factory, &[10, 2], vec![1.0; 20]).add(&tensor(factory, &[10, 2], vec![2.0; 20])),
        cpu_tensor(&[10, 2], vec![3.0; 20]),
    );
}

fn test_sub<T: Tensor>(factory: TensorFactory<T>) {
    // No batching: subtraction should mirror addition with the right operand negated.
    assert_output_eq(
        tensor(factory, &[2, 3], vec![
            11.0, 22.0, 33.0,
            44.0, 55.0, 66.0,
        ]).sub(&tensor(factory, &[2, 3], vec![
            10.0, 20.0, 30.0,
            40.0, 50.0, 60.0,
        ])),
        cpu_tensor(&[2, 3], vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]),
    );

    // Left-only batching: the unbatched right operand should subtract from every left batch.
    assert_output_eq(
        tensor(factory, &[2, 2, 2], vec![
            // batch 0
            1.0, 2.0,
            3.0, 4.0,
            // batch 1
            5.0, 6.0,
            7.0, 8.0,
        ]).sub(&tensor(factory, &[2, 2], vec![
            10.0, 20.0,
            30.0, 40.0,
        ])),
        cpu_tensor(&[2, 2, 2], vec![
            // batch 0
             -9.0, -18.0,
            -27.0, -36.0,
            // batch 1
             -5.0, -14.0,
            -23.0, -32.0,
        ]),
    );

    // Matching batch sizes: subtraction should pair batches one-to-one.
    assert_output_eq(
        tensor(factory, &[2, 2, 2], vec![
            // batch 0
            10.0, 20.0,
            30.0, 40.0,
            // batch 1
            50.0, 60.0,
            70.0, 80.0,
        ]).sub(&tensor(factory, &[2, 2, 2], vec![
            // batch 0
            1.0, 2.0,
            3.0, 4.0,
            // batch 1
            5.0, 6.0,
            7.0, 8.0,
        ])),
        cpu_tensor(&[2, 2, 2], vec![
            // batch 0
             9.0, 18.0,
            27.0, 36.0,
            // batch 1
            45.0, 54.0,
            63.0, 72.0,
        ]),
    );

    // Wide matrix: subtraction should stay element-wise over long rows.
    assert_output_eq(
        tensor(factory, &[2, 10], vec![3.0; 20]).sub(&tensor(factory, &[2, 10], vec![1.0; 20])),
        cpu_tensor(&[2, 10], vec![2.0; 20]),
    );

    // Tall matrix: subtraction should stay element-wise over many short rows.
    assert_output_eq(
        tensor(factory, &[10, 2], vec![3.0; 20]).sub(&tensor(factory, &[10, 2], vec![1.0; 20])),
        cpu_tensor(&[10, 2], vec![2.0; 20]),
    );
}

fn test_mul<T: Tensor>(factory: TensorFactory<T>) {
    // No batching: rectangular matrix multiplication should use row × column dot products.
    assert_output_eq(
        tensor(factory, &[2, 3], vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]).mul(&tensor(factory, &[3, 2], vec![
             7.0,  8.0,
             9.0, 10.0,
            11.0, 12.0,
        ])),
        cpu_tensor(&[2, 2], vec![
             58.0,  64.0,
            139.0, 154.0,
        ]),
    );

    // Left-only batching: one right matrix should multiply every left batch.
    assert_output_eq(
        tensor(factory, &[2, 2, 3], vec![
            // batch 0
            1.0,  2.0,  3.0,
            4.0,  5.0,  6.0,
            // batch 1
            7.0,  8.0,  9.0,
            10.0, 11.0, 12.0,
        ]).mul(&tensor(factory, &[3, 2], vec![
             7.0,  8.0,
             9.0, 10.0,
            11.0, 12.0,
        ])),
        cpu_tensor(&[2, 2, 2], vec![
            // batch 0
             58.0,  64.0,
            139.0, 154.0,
            // batch 1
            220.0, 244.0,
            301.0, 334.0,
        ]),
    );

    // Matching batch sizes: each left matrix should multiply the matching right matrix.
    assert_output_eq(
        tensor(factory, &[2, 2, 2], vec![
            // batch 0
            1.0, 2.0,
            3.0, 4.0,
            // batch 1
            5.0, 6.0,
            7.0, 8.0,
        ]).mul(&tensor(factory, &[2, 2, 2], vec![
            // batch 0: identity matrix
            1.0, 0.0,
            0.0, 1.0,
            // batch 1: row-sum matrix
            1.0, 1.0,
            1.0, 1.0,
        ])),
        cpu_tensor(&[2, 2, 2], vec![
            // batch 0
            1.0,  2.0,
            3.0,  4.0,
            // batch 1
            11.0, 11.0,
            15.0, 15.0,
        ]),
    );

    // Wide-left multiplication: verifies index math when width is much larger than height.
    assert_output_eq(
        tensor(factory, &[2, 10], vec![1.0; 20]).mul(&tensor(factory, &[10, 2], vec![1.0; 20])),
        cpu_tensor(&[2, 2], vec![10.0; 4]),
    );

    // Tall-left multiplication: verifies index math when height is much larger than width.
    assert_output_eq(
        tensor(factory, &[10, 2], vec![1.0; 20]).mul(&tensor(factory, &[2, 10], vec![1.0; 20])),
        cpu_tensor(&[10, 10], vec![2.0; 100]),
    );

    // Batched rectangular broadcast: output stride differs from the left matrix stride.
    assert_output_eq(
        tensor(factory, &[2, 2, 3], vec![1.0; 12]).mul(&tensor(factory, &[3, 4], vec![1.0; 12])),
        cpu_tensor(&[2, 2, 4], vec![3.0; 16]),
    );

    // Batched rectangular same-size: catches incorrect reuse of right batches.
    assert_output_eq(
        tensor(factory, &[2, 2, 3], vec![1.0; 12]).mul(&tensor(factory, &[2, 3, 4], vec![
            // batch 0
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            // batch 1
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
        ])),
        cpu_tensor(&[2, 2, 4], vec![
            // batch 0
            3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0,
            // batch 1
            6.0, 6.0, 6.0, 6.0,
            6.0, 6.0, 6.0, 6.0,
        ]),
    );
}

#[test]
fn test_cpu() {
    fn make_cpu(shape: TensorShape, data: Vec<f32>) -> CPUTensor {
        CPUTensor::from_data(shape, data).expect("test data should fill CPU tensor shape exactly")
    }

    test_relu::<CPUTensor>(make_cpu);
    test_neg::<CPUTensor>(make_cpu);
    test_add::<CPUTensor>(make_cpu);
    test_sub::<CPUTensor>(make_cpu);
    test_mul::<CPUTensor>(make_cpu);
}
