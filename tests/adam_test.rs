use llm::adam::Adam;
use ndarray::Array2;

#[test]
fn test_adam_initialization() {
    let shape = [2, 3];
    let adam = Adam::new((2, 3));

    // 动量矩阵与二阶矩矩阵初始时都应为 0。
    assert_eq!(adam.m.shape(), shape);
    assert_eq!(adam.v.shape(), shape);
    assert!(adam.m.iter().all(|&x| x == 0.0));
    assert!(adam.v.iter().all(|&x| x == 0.0));
}

#[test]
fn test_adam_step() {
    let shape = (2, 2);
    let lr = 0.001;
    let mut adam = Adam::new(shape);
    let mut params = Array2::ones(shape);
    let grads = Array2::ones(shape);

    // 记录初始参数。
    let initial_params = params.clone();

    // 执行一次优化步。
    adam.step(&mut params, &grads, lr);

    // 参数应发生变化。
    assert_ne!(params, initial_params);

    // 梯度为正时，参数应减小。
    assert!(params.iter().all(|&x| x < 1.0));
}

#[test]
fn test_adam_multiple_steps() {
    let shape = (2, 2);
    let lr = 0.001;
    let mut adam = Adam::new(shape);
    let mut params = Array2::ones(shape);
    let grads = Array2::ones(shape);

    // 记录初始参数。
    let initial_params = params.clone();

    // 连续执行多次优化步。
    for _ in 0..10 {
        adam.step(&mut params, &grads, lr);
    }

    // 参数变化应比单步更明显。
    assert!(params.iter().all(|&x| x < initial_params[[0, 0]]));
}

#[test]
fn test_adam_with_zero_gradients() {
    let shape = (2, 2);
    let lr = 0.001;
    let mut adam = Adam::new(shape);
    let mut params = Array2::ones(shape);
    let grads = Array2::zeros(shape);

    // 记录初始参数。
    let initial_params = params.clone();

    // 用全 0 梯度执行优化步。
    adam.step(&mut params, &grads, lr);

    // 梯度全 0 时参数不应变化。
    assert_eq!(params, initial_params);
}

#[test]
fn test_adam_with_negative_gradients() {
    let shape = (2, 2);
    let lr = 0.001;
    let mut adam = Adam::new(shape);
    let mut params = Array2::ones(shape);
    let grads = Array2::from_shape_fn(shape, |_| -1.0);

    // 执行一次优化步。
    adam.step(&mut params, &grads, lr);

    // 梯度为负时，参数应增大。
    assert!(params.iter().all(|&x| x > 1.0));
}
