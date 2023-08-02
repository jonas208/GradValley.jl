#=
This example was written by Jonas S. and is part of the GradValley.jl repository.

About:
This is the source of the ResNets model definitions found in the tutorials and examples section of the documentation: https://jonas208.github.io/GradValley.jl/tutorials_and_examples/
This script is also used by the pre-trained ResNets found in the pre-trained models section of the documentation: https://jonas208.github.io/GradValley.jl/(pre-trained)_models/#ResNet18/34/50/101/152-(Image-Classification)

Important Note:
It is heavily recommended to run this file (or the file in which you inlcude ResNets.jl), and any other files using GradValley, with multiple threads.
Using multiple threads can make training and calculating predictions much faster.
To do this, use the -t option when running a julia script in terminal/PowerShell/command line/etc.
If your CPU has 24 threads, for example, then run:
julia -t 24 ./ResNets.jl
The specified number of threads should match the number of threads your CPU provides.
=#
# import GradValley
using GradValley
using GradValley.Layers

# define a ResBlock (with optional downsampling) 
function ResBlock(in_channels::Int, out_channels::Int, downsample::Bool)
    # define modules
    if downsample
        shortcut = SequentialContainer([
            Conv(in_channels, out_channels, (1, 1), stride=(2, 2), use_bias=false),
            BatchNorm(out_channels)
        ])
        conv1 = Conv(in_channels, out_channels, (3, 3), stride=(2, 2), padding=(1, 1), use_bias=false)
    else
        shortcut = Identity()
        conv1 = Conv(in_channels, out_channels, (3, 3), padding=(1, 1), use_bias=false)
    end

    conv2 = Conv(out_channels, out_channels, (3, 3), padding=(1, 1), use_bias=false)
    bn1 = BatchNorm(out_channels, activation_function="relu")
    bn2 = BatchNorm(out_channels) # , activation_function="relu"

    relu = Identity(activation_function="relu")

    # define the forward pass with the residual/skipped connection
    function forward_pass(modules, input)
        # extract modules from modules vector (not necessary (therefore commented out) because the forward_pass function is defined in the ResBlock function (not somewhere "outside").)
        # shortcut, conv1, conv2, bn1, bn2, relu = modules
        # compute shortcut
        output_shortcut = forward(shortcut, input)
        # compute sequential part
        output = forward(bn1, forward(conv1, input))
        output = forward(bn2, forward(conv2, output))
        # residual/skipped connection
        output = forward(relu, output + output_shortcut) 
        
        return output
    end

    # initialize a container representing the ResBlock
    modules = [shortcut, conv1, conv2, bn1, bn2, relu]
    res_block = GraphContainer(forward_pass, modules)

    return res_block
end

# define a ResBottelneckBlock (with optional downsampling) 
function ResBottelneckBlock(in_channels::Int, out_channels::Int, downsample::Bool)
    # define modules
    shortcut = Identity()   

    if downsample || in_channels != out_channels
        if downsample
            shortcut = SequentialContainer([
                Conv(in_channels, out_channels, (1, 1), stride=(2, 2), use_bias=false),
                BatchNorm(out_channels)
            ])
        else
            shortcut = SequentialContainer([
                Conv(in_channels, out_channels, (1, 1), use_bias=false),
                BatchNorm(out_channels)
            ])
        end
    end

    conv1 = Conv(in_channels, out_channels ÷ 4, (1, 1), use_bias=false)
    if downsample
        conv2 = Conv(out_channels ÷ 4, out_channels ÷ 4, (3, 3), stride=(2, 2), padding=(1, 1), use_bias=false)
    else
        conv2 = Conv(out_channels ÷ 4, out_channels ÷ 4, (3, 3), padding=(1, 1), use_bias=false)
    end
    conv3 = Conv(out_channels ÷ 4, out_channels, (1, 1), use_bias=false)

    bn1 = BatchNorm(out_channels ÷ 4, activation_function="relu")
    bn2 = BatchNorm(out_channels ÷ 4, activation_function="relu")
    bn3 = BatchNorm(out_channels) # , activation_function="relu"

    relu = Identity(activation_function="relu")

    # define the forward pass with the residual/skipped connection
    function forward_pass(modules, input)
        # extract modules from modules vector (not necessary (therefore commented out) because the forward_pass function is defined in the ResBlock function (not somewhere "outside").)
        # shortcut, conv1, conv2, conv3, bn1, bn2, bn3, relu = modules
        # compute shortcut
        output_shortcut = forward(shortcut, input)
        # compute sequential part
        output = forward(bn1, forward(conv1, input))
        output = forward(bn2, forward(conv2, output))
        output = forward(bn3, forward(conv3, output))
        # residual/skipped connection
        output = forward(relu, output + output_shortcut) 
        
        return output
    end

    # initialize a container representing the ResBlock
    modules = [shortcut, conv1, conv2, conv3, bn1, bn2, bn3, relu]
    res_bottelneck_block = GraphContainer(forward_pass, modules)

    return res_bottelneck_block
end

# define a ResNet 
function ResNet(in_channels::Int, ResBlock::Union{Function, DataType}, repeat::Vector{Int}; use_bottelneck::Bool=false, classes::Int=1000)
    # define layer0
    layer0 = SequentialContainer([
        Conv(in_channels, 64, (7, 7), stride=(2, 2), padding=(3, 3), use_bias=false),
        BatchNorm(64, activation_function="relu"),
        MaxPool((3, 3), stride=(2, 2), padding=(1, 1))
    ])

    # define number of filters/channels
    if use_bottelneck
        filters = Int[64, 256, 512, 1024, 2048]
    else
        filters = Int[64, 64, 128, 256, 512]
    end

    # define the following modules
    layer1_modules = [ResBlock(filters[1], filters[2], false)]
    for i in 1:repeat[1] - 1
        push!(layer1_modules, ResBlock(filters[2], filters[2], false))
    end
    layer1 = SequentialContainer(layer1_modules)

    layer2_modules = [ResBlock(filters[2], filters[3], true)]
    for i in 1:repeat[2] - 1
        push!(layer2_modules, ResBlock(filters[3], filters[3], false))
    end
    layer2 = SequentialContainer(layer2_modules)

    layer3_modules = [ResBlock(filters[3], filters[4], true)]
    for i in 1:repeat[3] - 1
        push!(layer3_modules, ResBlock(filters[4], filters[4], false))
    end
    layer3 = SequentialContainer(layer3_modules)

    layer4_modules = [ResBlock(filters[4], filters[5], true)]
    for i in 1:repeat[4] - 1
        push!(layer4_modules, ResBlock(filters[5], filters[5], false))
    end
    layer4 = SequentialContainer(layer4_modules)

    gap = AdaptiveAvgPool((1, 1))
    flatten = Reshape((filters[5], ))
    fc = Fc(filters[5], classes)

    # initialize a container representing the ResNet
    res_net = SequentialContainer([layer0, layer1, layer2, layer3, layer4, gap, flatten, fc])

    return res_net
end

# construct a ResNet18
function ResNet18(in_channels=3, classes=1000)
    return ResNet(in_channels, ResBlock, [2, 2, 2, 2], use_bottelneck=false, classes=classes)
end

# construct a ResNet34
function ResNet34(in_channels=3, classes=1000)
    return ResNet(in_channels, ResBlock, [3, 4, 6, 3], use_bottelneck=false, classes=classes)
end

# construct a ResNet50
function ResNet50(in_channels=3, classes=1000)
    return ResNet(in_channels, ResBottelneckBlock, [3, 4, 6, 3], use_bottelneck=true, classes=classes)
end

# construct a ResNet101
function ResNet101(in_channels=3, classes=1000)
    return ResNet(in_channels, ResBottelneckBlock, [3, 4, 23, 3], use_bottelneck=true, classes=classes)
end

# construct a ResNet152
function ResNet152(in_channels=3, classes=1000)
    return ResNet(in_channels, ResBottelneckBlock, [3, 8, 36, 3], use_bottelneck=true, classes=classes)
end