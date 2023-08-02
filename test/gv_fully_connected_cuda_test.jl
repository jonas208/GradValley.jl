in_features, out_features = 128, 64
# in_features, out_features = 4000, 1000
dtype = Float64

input_cpu = rand(dtype, in_features, 64)
weight_cpu = rand(dtype, out_features, in_features)
bias_cpu = rand(dtype, out_features)

input_gpu = CuArray(input_cpu)
weight_gpu = CuArray(weight_cpu)
bias_gpu = CuArray(bias_cpu)

output_cpu = fc_forward(input_cpu, weight_cpu, bias_cpu)
output_gpu = fc_forward(input_gpu, weight_gpu, bias_gpu)

output_gradient_cpu = rand(dtype, size(output_cpu)...)
output_gradient_gpu = CuArray(output_gradient_cpu)

input_gradient_cpu, weight_gradient_cpu, bias_gradient_cpu = fc_backward(output_gradient_cpu, input_cpu, weight_cpu)
input_gradient_gpu, weight_gradient_gpu, bias_gradient_gpu = fc_backward(output_gradient_gpu, input_gpu, weight_gpu)

@test isapprox(CuArray(output_cpu), output_gpu)
@test isapprox(CuArray(input_gradient_cpu), input_gradient_gpu)
@test isapprox(CuArray(weight_gradient_cpu), weight_gradient_gpu)
@test isapprox(CuArray(bias_gradient_cpu), bias_gradient_gpu)