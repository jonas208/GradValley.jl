using GradValley
using GradValley.Layers
using GradValley.Optimization
using CUDA
using FileIO

# Load the preprocessed data
batches = load("CelebA-HQ_preprocessed.jld2", "data")

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
# e.g. 25 epochs on Float32 for both GPU and CPU or 10 epochs on Float64 for the CPU
num_epochs = 25

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# eltype of data and parameters
# Float32 or Float64
dtype = Float32

generator = SequentialContainer([
    # input is Z, going into a convolution
    ConvTranspose(nz, ngf * 8, (4, 4), stride=(1, 1), padding=(0, 0), use_bias=false),
    BatchNorm(ngf * 8, activation_function="relu"),
    # state size. (ngf*8) x 4 x 4
    ConvTranspose(ngf * 8, ngf * 4, (4, 4), stride=(2, 2), padding=(1, 1), use_bias=false),
    BatchNorm(ngf * 4, activation_function="relu"),
    # state size. (ngf*4) x 8 x 8
    ConvTranspose(ngf * 4, ngf * 2, (4, 4), stride=(2, 2), padding=(1, 1), use_bias=false),
    BatchNorm(ngf * 2, activation_function="relu"),
    # state size. (ngf*2) x 16 x 16
    ConvTranspose(ngf * 2, ngf, (4, 4), stride=(2, 2), padding=(1, 1), use_bias=false),
    BatchNorm(ngf, activation_function="relu"),
    # state size. (ngf) x 32 x 32
    ConvTranspose(ngf, nc, (4, 4), stride=(2, 2), padding=(1, 1), use_bias=false, activation_function="tanh")
    # state size. (nc) x 64 x 64
])

discriminator = SequentialContainer([
    # input is (nc) x 64 x 64
    Conv(nc, ndf, (4, 4), stride=(2, 2), padding=(1, 1), use_bias=false, activation_function="leaky_relu:0.2"),
    # state size. (ndf) x 32 x 32
    Conv(ndf, ndf * 2, (4, 4), stride=(2, 2), padding=(1, 1), use_bias=false),
    BatchNorm(ndf * 2, activation_function="leaky_relu:0.2"),
    # state size. (ndf*2) x 16 x 16
    Conv(ndf * 2, ndf * 4, (4, 4), stride=(2, 2), padding=(1, 1), use_bias=false),
    BatchNorm(ndf * 4, activation_function="leaky_relu:0.2"),
    # state size. (ndf*4) x 8 x 8
    Conv(ndf * 4, ndf * 8, (4, 4), stride=(2, 2), padding=(1, 1), use_bias=false),
    BatchNorm(ndf * 8, activation_function="leaky_relu:0.2"),
    # state size. (ndf*8) x 4 x 4
    Conv(ndf * 8, 1, (4, 4), stride=(1, 1), padding=(0, 0), use_bias=false, activation_function="sigmoid")
])

# check if CUDA is available
use_cuda = CUDA.functional()
# move the model to the correct device and convert its parameters to the specified dtype
if use_cuda
    println("The GPU is used")
    module_to_eltype_device!(generator, element_type=dtype, device="gpu")
    module_to_eltype_device!(discriminator, element_type=dtype, device="gpu")
else
    println("The CPU is used")
    module_to_eltype_device!(generator, element_type=dtype, device="cpu")
    module_to_eltype_device!(discriminator, element_type=dtype, device="cpu")
end

# Setup the loss function
criterion = bce_loss

# Create batch of latent vectors that we will use to visualize the progression of the generator
if use_cuda
    fixed_noise = CUDA.randn(dtype, 1, 1, nz, 64)
else
    fixed_noise = randn(dtype, 1, 1, nz, 64)
end 

# Establish convention for real and fake labels during training
real_label = dtype(1)
fake_label = dtype(0)

# Setup Adam optimizers for both G and D
optimizerD = Adam(discriminator, learning_rate=lr, beta1=beta1, beta2=0.999)
optimizerG = Adam(generator, learning_rate=lr, beta1=beta1, beta2=0.999)

function train()
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    global iters = 0

    println("Starting Training Loop...")
    # For each epoch
    for epoch in 1:num_epochs

        # save some interim results when using the CPU
        if !use_cuda && epoch == 6
            file_name_img_list = "img_list_intermediate_result.jld2"
            save(file_name_img_list, Dict("img_list" => img_list))
        end

        # For each batch in the data
        for (i, batch) in enumerate(batches)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            zero_gradients(discriminator)

            if eltype(batch) != dtype
                batch = convert(Array{dtype, 4}, batch)
            end

            # Format batch
            if use_cuda
                real = CuArray(batch)
            else
                real = batch
            end

            b_size = size(real)[end]
            if use_cuda
                label = CUDA.fill(real_label, (1, 1, 1, b_size))
            else
                label = fill(real_label, (1, 1, 1, b_size))
            end
            # Forward pass real batch through D
            output = forward(discriminator, real)
            # Calculate loss on all-real batch
            errD_real, errD_real_derivative = criterion(output, label)
            # Calculate gradients for D in backward pass
            backward(discriminator, errD_real_derivative)
            D_x = sum(output) / length(output)

            ## Train with all-fake batch
            # Generate batch of latent vectors
            if use_cuda
                noise = CUDA.randn(dtype, 1, 1, nz, b_size)
            else
                noise = randn(dtype, 1, 1, nz, b_size)
            end
            # Generate fake image batch with G
            fake = forward(generator, noise)
            if use_cuda
                CUDA.fill!(label, fake_label)
            else
                fill!(label, fake_label)
            end
            # Classify all fake batch with D
            output = forward(discriminator, fake)
            # Calculate D's loss on the all-fake batch
            errD_fake, errD_fake_derivative = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            backward(discriminator, errD_fake_derivative)
            D_G_z1 = sum(output) / length(output)
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            step!(optimizerD)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            zero_gradients(generator)
            # fake labels are real for generator cost
            if use_cuda
                CUDA.fill!(label, real_label) 
            else
                fill!(label, real_label) 
            end
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = forward(discriminator, fake)
            # Calculate G's loss based on this output
            errG, errG_derivative = criterion(output, label)
            # Calculate gradients for G
            # The gradient flow does not reach the generator automatically, 
            # so we have to do that manually by passing the gradient returned from the backward pass of the discriminator as the derivative_loss input to the generator backward call
            input_gradient = backward(discriminator, errG_derivative)
            backward(generator, input_gradient)
            D_G_z2 = sum(output) / length(output)
            # Update G
            step!(optimizerG)

            # Output training stats
            if i % 1 == 0 # i % 50 == 0
                println("[$epoch/$num_epochs][$i/$(235)]\tLoss_D: $(round(errD, digits=4))\tLoss_G: $(round(errG, digits=4))\tD(x): $(round(D_x, digits=4))\tD(G(z)): $(round(D_G_z1, digits=4)) / $(round(D_G_z2, digits=4))")
            end

            # Save Losses for potential plotting later
            push!(G_losses, errG)
            push!(D_losses, errD)

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 50 == 0) || ((epoch == num_epochs) && (i == length(batches)))
                # testmode!(generator)
                fake = forward(generator, fixed_noise)
                # trainmode!(generator)
                push!(img_list, fake)
            end

            global iters += 1
        end
    end

    return img_list, G_losses, D_losses
end

# Start training 
if use_cuda
    img_list, G_losses, D_losses = CUDA.@time train()
else
    img_list, G_losses, D_losses = @time train()
end

# move the intermediate results on fixed_noise in img_list and the models back to the CPU for saving
if use_cuda
    for i in eachindex(img_list)
        img_list[i] = convert(Array{dtype, 4}, img_list[i])
    end
    # note that clean_model_from_backward_information! runs automatically in the background when calling module_to_eltype_device! on a container
    module_to_eltype_device!(discriminator, element_type=dtype, device="cpu")
    module_to_eltype_device!(generator, element_type=dtype, device="cpu")
end

# Save the models 
file_nameD = "discriminator.jld2"
save(file_nameD, Dict("discriminator" => discriminator))
file_nameG = "generator.jld2"
save(file_nameG, Dict("generator" => generator))
# Save the image list (intermediate results on fixed_noise)
file_name_img_list = "img_list.jld2"
save(file_name_img_list, Dict("img_list" => img_list))