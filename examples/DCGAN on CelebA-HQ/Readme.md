# Deep Convolutional Generative Adversarial Network (DCGAN) on CelebA-HQ

This is the code used by the [DCGAN Tutorial](https://jonas208.github.io/GradValley.jl/dev/tutorials_and_examples/#Deep-Convolutional-Generative-Adversarial-Network-(DCGAN)-on-CelebA-HQ) in the [documentation](https://jonas208.github.io/GradValley.jl/dev/).

This example/tutorial can be seen as a reimplementation of [PyTorch's DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) with the difference
that we are using CelebA-HQ (approx. 30,000 images) here instead of the normal CelebA (approx. 200,000 images) dataset. 
Note that this tutorial doesn't cover the theory behind DCGANs, it just focuses on the implementation in Julia with GradValley.jl.
You can find detailed information about the theory and a step by step implementation in the awesome [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).

You can find some [example results](https://jonas208.github.io/GradValley.jl/dev/tutorials_and_examples/#Results-2) in the [folder](https://github.com/jonas208/GradValley.jl/tree/main/examples/DCGAN%20on%20CelebA-HQ/example_results) 
of the same name and in the [tutorial](https://jonas208.github.io/GradValley.jl/dev/tutorials_and_examples/#Results-2) in the documentation.

## Structure and order of execution
- 1. `preprocess_data.jl` for preprocessing the training data (this file uses `preprocessing_for_resnets.jl`)
- 2. `DCGAN.jl` for training the DCGAN (runs on GPU or CPU, run with multiple threads if you only have a CPU!)
- 3. `show_img_list.jl` for showing the model's results on fixed noise
- 4. `inference.jl` for running inference (generates images using the trained generator model and save them as image files)
 
For more information, please see the [tutorial](https://jonas208.github.io/GradValley.jl/dev/tutorials_and_examples/#Deep-Convolutional-Generative-Adversarial-Network-(DCGAN)-on-CelebA-HQ) in the documentation.
