out_shape = (50 * 50, )
input_cpu = rand(dtype, 50, 50, 64)
input_gpu = CuArray(input_cpu)
dtype = Float64

output_cpu = reshape_forward(input_cpu, out_shape)
output_gpu = reshape_forward(input_gpu, out_shape)

output_gradient_cpu = rand(dtype, size(output_cpu)...)
output_gradient_gpu = CuArray(output_gradient_cpu)

input_gradient_cpu = reshape_backward(output_gradient_cpu, input_cpu)
input_gradient_gpu = reshape_backward(output_gradient_gpu, input_gpu)

@test isapprox(CuArray(output_cpu), output_gpu)
@test isapprox(CuArray(input_gradient_cpu), input_gradient_gpu)