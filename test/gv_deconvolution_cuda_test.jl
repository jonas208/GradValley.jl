stride = (2, 1)
padding = (1, 0)
output_padding = (1, 0)
dilation = (2, 1)
groups = 2
dtype = Float64

input_cpu = rand(dtype, 150, 150, 4, 64)
weight_cpu = rand(dtype, 3, 3, Int(20/groups), 4)
bias_cpu = rand(dtype, 20)

input_gpu = CuArray(input_cpu)
weight_gpu = CuArray(weight_cpu)
bias_gpu = CuArray(bias_cpu)

output_cpu = deconvolution2d(input_cpu, weight_cpu, bias_cpu, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups)
output_gpu = deconvolution2d(input_gpu, weight_gpu, bias_gpu, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups)

output_gradient_cpu = rand(dtype, size(output_cpu)...)
output_gradient_gpu = CuArray(output_gradient_cpu)

input_gradient_cpu = deconvolution2d_data_backward(output_gradient_cpu, input_cpu, weight_cpu, stride=stride, padding=padding, dilation=dilation, groups=groups)
input_gradient_gpu = deconvolution2d_data_backward(output_gradient_gpu, input_gpu, weight_gpu, stride=stride, padding=padding, dilation=dilation, groups=groups)

weight_gradient_cpu = deconvolution2d_filter_backward(output_gradient_cpu, input_cpu, weight_cpu, stride=stride, padding=padding, dilation=dilation, groups=groups)
weight_gradient_gpu = deconvolution2d_filter_backward(output_gradient_gpu, input_gpu, weight_gpu, stride=stride, padding=padding, dilation=dilation, groups=groups)

bias_gradient_cpu = deconvolution2d_bias_backward(output_gradient_cpu)
bias_gradient_gpu = deconvolution2d_bias_backward(output_gradient_gpu)

@test isapprox(CuArray(output_cpu), output_gpu)
@test isapprox(CuArray(input_gradient_cpu), input_gradient_gpu)
@test isapprox(CuArray(weight_gradient_cpu), weight_gradient_gpu)
@test isapprox(CuArray(bias_gradient_cpu), bias_gradient_gpu)