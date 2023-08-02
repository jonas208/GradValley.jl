num_features = 4
input_cpu = rand(dtype, 150, 150, num_features, 64)
input_gpu = CuArray(input_cpu)
dtype = Float64

# initialize weights
weight_gamma_cpu = rand(dtype, num_features) # ones(dtype, num_features)
weight_beta_cpu = rand(dtype, num_features) # zeros(dtype, num_features)

# initialize running statistics
running_mean_cpu = zeros(dtype, num_features)
running_variance_cpu = ones(dtype, num_features)

weight_gamma_gpu = CuArray(weight_gamma_cpu)
weight_beta_gpu = CuArray(weight_beta_cpu)

running_mean_gpu = CuArray(running_mean_cpu)
running_variance_gpu = CuArray(running_variance_cpu)

test_mode = false
track_running_stats = true

momentum = dtype(0.1)
epsilon = dtype(1e-5)

output_cpu, new_running_mean_cpu, new_running_variance_cpu = batch_norm2d_forward(input_cpu, weight_gamma_cpu, weight_beta_cpu, track_running_stats, running_mean_cpu, running_variance_cpu, test_mode, momentum=momentum, epsilon=epsilon)
output_gpu, new_running_mean_gpu, new_running_variance_gpu = batch_norm2d_forward(input_gpu, weight_gamma_gpu, weight_beta_gpu, track_running_stats, running_mean_gpu, running_variance_gpu, test_mode, momentum=momentum, epsilon=epsilon)

@test isapprox(CuArray(running_mean_cpu), running_mean_gpu)
@test isapprox(CuArray(running_variance_cpu), running_variance_gpu)

@test isapprox(CuArray(output_cpu), output_gpu)
@test isapprox(CuArray(new_running_mean_cpu), new_running_mean_gpu)
@test isapprox(CuArray(new_running_variance_cpu), new_running_variance_gpu)

output_gradient_cpu = rand(dtype, size(output_cpu)...)
output_gradient_gpu = CuArray(output_gradient_cpu)

input_gradient_cpu = batch_norm2d_data_backward(output_gradient_cpu, output_cpu, input_cpu, weight_gamma_cpu, weight_beta_cpu, track_running_stats, new_running_mean_cpu, new_running_variance_cpu, test_mode, epsilon=epsilon)
input_gradient_gpu = batch_norm2d_data_backward(output_gradient_gpu, output_gpu, input_gpu, weight_gamma_gpu, weight_beta_gpu, track_running_stats, new_running_mean_gpu, new_running_variance_gpu, test_mode, epsilon=epsilon)

weight_gamma_gradient_cpu, weight_beta_gradient_cpu = batch_norm2d_weight_backward(output_gradient_cpu, output_cpu, weight_gamma_cpu, weight_beta_cpu)
weight_gamma_gradient_gpu, weight_beta_gradient_gpu = batch_norm2d_weight_backward(output_gradient_gpu, output_gpu, input_gpu, weight_gamma_gpu, weight_beta_gpu, track_running_stats, new_running_mean_gpu, new_running_variance_gpu, test_mode, epsilon=epsilon)

@test isapprox(CuArray(input_gradient_cpu), input_gradient_gpu)
@test isapprox(CuArray(weight_gamma_gradient_cpu), weight_gamma_gradient_gpu)
@test isapprox(CuArray(weight_beta_gradient_cpu), weight_beta_gradient_gpu)

input_gradient_cpu, weight_gamma_gradient_cpu, weight_beta_gradient_cpu = batch_norm2d_backward(output_gradient_cpu, output_cpu, input_cpu, weight_gamma_cpu, weight_beta_cpu, track_running_stats, new_running_mean_cpu, new_running_variance_cpu, test_mode, epsilon=epsilon)
input_gradient_gpu, weight_gamma_gradient_gpu, weight_beta_gradient_gpu = batch_norm2d_backward(output_gradient_gpu, output_gpu, input_gpu, weight_gamma_gpu, weight_beta_gpu, track_running_stats, new_running_mean_gpu, new_running_variance_gpu, test_mode, epsilon=epsilon)

@test isapprox(CuArray(input_gradient_cpu), input_gradient_gpu)
@test isapprox(CuArray(weight_gamma_gradient_cpu), weight_gamma_gradient_gpu)
@test isapprox(CuArray(weight_beta_gradient_cpu), weight_beta_gradient_gpu)