# include("GradValley/GradValley.jl")
# using .GradValley, .GradValley.Layers, .GradValley.Optimization, .GradValley.Functional
using GradValley, GradValley.Layers, GradValley.Optimization, GradValley.Functional
using Test

"""
test_layer_initializations()

Test the initialization of weights, activation functions, the initialization of the SequentialContainer and the summarize_model function.
Note that this model makes absolutely no sense, it is only intended to test as many layers as possible.
"""
function test_layer_initializations()
    # test each initialization steps of all available layers
    conv = Conv(3, 6, (5, 5), stride=(1, 2), padding=(0, 1), dilation=(0, 2), groups=3, activation_function="relu", init_mode="default_uniform", use_bias=true)
    depthwise_conv = DepthwiseConv(6, 12, (5, 5), stride=(1, 2), padding=(0, 1), dilation=(0, 2), activation_function="tanh", init_mode="default", use_bias=true)
    conv_transpose = ConvTranspose(12, 6, (5, 5), stride=(1, 2), padding=(0, 1), dilation=(0, 2), output_padding=(0, 1), groups=2)
    identity = Identity(activation_function="sigmoid")
    batch_norm = BatchNorm2d(6, epsilon=1e-05, momentum=0.1, affine=true, track_running_stats=true, activation_function=nothing)
    max_pool = MaxPool((2, 2), stride=(1, 2), padding=(0, 1), dilation=(0, 2), activation_function="sigmoid")
    avg_pool = AvgPool((2, 2), stride=(1, 2), padding=(0, 1), dilation=(0, 2), activation_function=nothing)
    adaptive_max_pool = AdaptiveMaxPool((50, 50), activation_function=nothing)
    adaptive_avg_pool = AdaptiveAvgPool((5, 5), activation_function=nothing)
    flatten = Reshape((150, )) # features = C*H*W = 12*5*5 = 300
    fully_connected_1 = Fc(150, 150, activation_function=nothing, init_mode="kaiming_uniform") # in_features = C*H*W = 12*5*5 = 300
    fully_connected_2 = Fc(150, 75, activation_function=nothing, init_mode="kaiming")
    fully_connected_3 = Fc(75, 38, activation_function=nothing, init_mode="xavier_uniform")
    softmax = Softmax(dim=2)
    fully_connected_4 = Fc(38, 20, activation_function=nothing, init_mode="xavier")
    # test initialization of SequentialContainers
    submodel_convs = SequentialContainer([conv, depthwise_conv, conv_transpose, identity])
    submodel_pools = SequentialContainer([max_pool, avg_pool, adaptive_max_pool, adaptive_avg_pool])
    submodel_feature_extractor = SequentialContainer([submodel_convs, batch_norm, submodel_pools, flatten])
    submodel_fcs = SequentialContainer([fully_connected_1, fully_connected_2, fully_connected_3, softmax, fully_connected_4])
    sc = SequentialContainer([submodel_feature_extractor, submodel_fcs])
    # test initialization of GraphContainers
    f(layers, x, y) = (1 .- (2.5 .+ (5 ./ (10 .* forward(layers[1], x) .+ y .- y .* y ./ y .+ 5 .- 2.5)))) # .* 10 .^ 2
    # f(layers, x, y) = forward(layers[1], x) .* 10 
    model = GraphContainer(f, [sc])
    # test the summary function
    println(summarize_model(model)[1])

    return model
end

"""
test_data_loader_initialization()

Test the initialization of a data loader.
Note that this data loader makes absolutely no sense, it is only intended to test as many functionalities as possible.
"""
function test_data_loader_initialization()
    get_function(index) = rand(3, 50, 50), rand(20)
    data_loader = DataLoader(get_function, 100, batch_size=4, shuffle=true, drop_last=true) # or dataset_size=1000 (or similar, but not divisible by batch_size for drop_last)
    # test index functionalities
    data = data_loader[3]
    data = data_loader[begin:end]
    println(size(data_loader))
    println(length(data_loader))

    return data_loader
end

"""
test_optimizer_initializations()

Test the initialization of different optimizers.
Note that these optimizers makes absolutely no sense, they are only intended to test as many functionalities as possible.
"""
function test_optimizer_initializations(model)
    optim_1 = SGD(model, 0.01; weight_decay=0.1, dampening=0.1, maximize=false)
    optim_2 = MSGD(model, 0.01; momentum=0.90, weight_decay=0.00, dampening=0.10, maximize=true)
    optim_3 = Nesterov(model, 0.01; momentum=0.90, weight_decay=0.1, dampening=0.00, maximize=false)

    return optim_1, optim_2, optim_3
end

"""
simulate_training()

Simulate a training for testing.
Note that this training makes absolutely no sense, it is only intended to test as many functionalities as possible.
"""
function simulate_training(model, data_loader, optim_1, optim_2, optim_3)
    for (batch, (images_batch, targets_batch)) in enumerate(data_loader)
        # test forward pass (only affects batchnorm: in training mode and in test mode)
        testmode!(model)
        predictions_batch_test_mode = forward(model, images_batch, rand(size(images_batch)[1], 20))
        trainmode!(model)
        predicitions_batch = forward(model, images_batch, rand(size(images_batch)[1], 20))
        zero_gradients(model)
        # test loss functions
        loss, derivative_loss = mae_loss(predicitions_batch, targets_batch)
        loss, derivative_loss = mse_loss(predicitions_batch, targets_batch)
        # test backward pass
        backward(model, derivative_loss)
        # test step functions with all optimizers
        step!(optim_1)
        step!(optim_2)
        step!(optim_3)
        # print status
        println(batch)
    end
end

# run all test functions
model = test_layer_initializations()
data_loader = test_data_loader_initialization()
optim_1, optim_2, optim_3 = test_optimizer_initializations(model)
simulate_training(model, data_loader, optim_1, optim_2, optim_3)


