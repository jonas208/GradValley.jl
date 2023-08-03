module GradValley
using Random

# including all necessary files
include("gv_functional.jl")
include("gv_layers.jl")
include("gv_optimization.jl")

# export utilities
export DataLoader, reshuffle!

# create batches of indices for a hole dataset (instead of loading the entire data set directly -> more memory efficient)
function load_indices_batches(dataset_size::Integer; batch_size::Integer=1, shuffle::Bool=false, drop_last::Bool=false)
    size_last_batch = dataset_size % batch_size
    if size_last_batch == 0
        size_last_batch = batch_size
        num_batches = convert(Int, dataset_size / batch_size)
    else
        num_batches = convert(Int, floor(dataset_size / batch_size)) + 1
    end

    if shuffle
        indices = randperm(dataset_size)
    else
        indices = Int[i for i in 1:dataset_size]
    end

    indices_batches = Vector{Vector{<: Integer}}(undef, num_batches)
    last_index = 0
    for index_batch in 1:num_batches
        current_batch_size = if index_batch != num_batches batch_size else size_last_batch end
        batch = Int[indices[last_index + index] for index in 1:current_batch_size]
        indices_batches[index_batch] = batch
        last_index += current_batch_size
    end

    if drop_last && size_last_batch != batch_size
        pop!(indices_batches)
    end

    return indices_batches
end

# loads one data batch, the get function and the indices batch are given
function load_data_batch(get_function::Function, indices_batch::Vector{<: Integer})
    current_batch_size = length(indices_batch)
    # prepare the batch arrays with their correct types and sizes
    first_element::Tuple = get_function(indices_batch[1])
    data_batch = []
    for array::AbstractArray in first_element
        num_dims = length(size(array))+1 # +1 for batch dimension
        # creates an array for one data batch with the element type of the given array from the get function
        batch = Array{eltype(array), num_dims}(undef, size(array)..., current_batch_size)
        push!(data_batch, batch)
    end
    data_batch = Tuple(data_batch)
    # fill the batches in the tuple with data
    for (i, index) in enumerate(indices_batch)
        element = get_function(index)
        for (j, batch) in enumerate(data_batch)
            ranges = [Colon() for _ in 1:ndims(batch)-1] # useful for indexing an array with arbitrary size
            batch[ranges..., i] = element[j]
        end
    end
    
    return data_batch
end

@doc raw"""
    DataLoader(get_function::Function, dataset_size::Integer; batch_size::Integer=1, shuffle::Bool=false, drop_last::Bool=false)

The DataLoader was designed to easily iterate over batches. Each time a new batch is requested, the data loader loads this batch "just in time" (instead of loading all the batches to memory at once). 

The `get_function` is expected to load one item from a dataset at a given index.
The specified `get_function` is expected to accept exactly one positional argument, which is the index of the item the `get_function` will return.
A tuple of arbitrary length is expected as the return value of the `get_function`. Each element in this tuple must be an array. The length/size and type of the tuple and array is expected to be the same at each index.
When a batch is requested, the data loader returns the tuple containing the with batch dimensions extended arrays.

!!! note
    The DataLoader is iteratabel and indexable. size(dataloader) returns the given size of the dataset, length(dataloader) returns the total number of batches (equal if batch_size=1).
    When a range is given as the index argument, a vector containing multiple batches (arrays) is returned.

!!! tip
    If you *really* want to load the whole dataset to memory (e.g. useful when training over multiple epochs, with this way, you don't have to reload the dataset each epoch over and over again), you can do so of course: `all_batches = dataloader[start:end]` where `typeof(dataloader) == DataLoader`

# Arguments
- `get_function::Function`: the function which takes the index of an item from a dataset and returns that item (an arbitrary sized tuple containing arrays)
- `dataset_size::Integer`: the maximum index the `get_function` accepts (the number of items in the dataset, the dataset size)
- `batch_size::Integer=1`: the batch size (the last dimension, the extended batch dimension, of each array in the returned tuple has this size)
- `shuffle::Bool=false`: reshuffle the data (doesn't reshuffle automatically after each epoch, use [`reshuffle!`](@ref) instead)
- `drop_last::Bool=false`: set to true to drop the last incomplete batch, if the dataset size is not divisible by the batch size, if false and the size of dataset is not divisible by the batch size, then the last batch will be smaller

# Examples
```julia-repl
# EXAMPLE FROM https://jonas208.github.io/GradValley.jl/tutorials_and_examples/#Tutorials-and-Examples
julia> using MLDatasets # a package for downloading datasets
# initialize train- and test-dataset
julia> mnist_train = MNIST(:train) 
julia> mnist_test = MNIST(:test)
# define the get_element function:
# function for getting an image and the corresponding target vector from the train or test partition
julia> function get_element(index, partition)
            # load one image and the corresponding label
            if partition == "train"
                image, label = mnist_train[index]
            else # test partition
                image, label = mnist_test[index]
            end
            # add channel dimension and rescaling the values to their original 8 bit gray scale values
            image = reshape(image, 28, 28, 1) .* 255
            # generate the target vector from the label, one for the correct digit, zeros for the wrong digits
            target = zeros(10)
            target[label + 1] = 1.00

            return image, target
       end
# initialize the data loaders (with anonymous function which helps to easily distinguish between test- and train-partition)
train_data_loader = DataLoader(index -> get_element(index, "train"), length(mnist_train), batch_size=32, shuffle=true)
test_data_loader = DataLoader(index -> get_element(index, "test"), length(mnist_test), batch_size=32)
# in most cases NOT recommended: you can force the data loaders to load all the data at once into memory, depending on the dataset's size, this may take a while
julia> # train_data = train_data_loader[begin:end] # turned off to save time
julia> # test_data = test_data_loader[begin:end] # turned off to save time
# now you can write your train- or test-loop like so 
julia> for (batch, (image_batch, target_batch)) in enumerate(test_data_loader) #=do anything useful here=# end
julia> for (batch, (image_batch, target_batch)) in enumerate(train_data_loader) #=do anything useful here=# end
```
"""
mutable struct DataLoader
    # characteristics of the dataloader
    get_function::Function
    dataset_size::Integer
    num_batches::Integer
    batch_size::Integer
    shuffle::Bool
    drop_last::Bool
    # data
    indices_batches::Vector{Vector{<: Integer}}
    # custom constructor
    function DataLoader(get_function::Function, dataset_size::Integer; batch_size::Integer=1, shuffle::Bool=false, drop_last::Bool=false)
        # create indices for all batches
        indices_batches = load_indices_batches(dataset_size, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        num_batches = length(indices_batches)

        # create new instance/object
        new(get_function,
            dataset_size,
            num_batches,
            batch_size,
            shuffle,
            drop_last,
            indices_batches
        )
    end
end

# making the DataLoader iterable
Base.iterate(DL::DataLoader, state=1) = state > DL.num_batches ? nothing : (load_data_batch(DL.get_function, DL.indices_batches[state]), state+1)
# making the length/size (=num_batches) of the DataLoader available
# length(DL) returns the number of batches (NOT equal to the dataset_size when batch_size != 1)
Base.length(DL::DataLoader) = DL.num_batches
# size(DL) returns the size of the dataset that was specified when the data loader was created (NOT equal to the number of batches for batch_size != 1)
Base.size(DL::DataLoader) = DL.dataset_size # DL.num_batches
# making the DataLoader indexable
function Base.getindex(DL::DataLoader, index::Integer)
    1 <= index <= DL.num_batches || throw(BoundsError(DL, index))
    return load_data_batch(DL.get_function, DL.indices_batches[index])
end
Base.firstindex(DL::DataLoader) = 1
Base.lastindex(DL::DataLoader) = DL.num_batches
function Base.getindex(DL::DataLoader, index_range::UnitRange{<: Integer})
    1 <= index_range[1] <= DL.num_batches || throw(BoundsError(DL, index))
    1 <= index_range[end] <= DL.num_batches || throw(BoundsError(DL, index))
    batches = [load_data_batch(DL.get_function, DL.indices_batches[index]) for index in index_range]

    return batches
end
# pretty-printing for the DataLoader struct
Base.show(io::IO, DL::DataLoader) = print(io, "DataLoader(get_function=$(DL.get_function), dataset_size=$(DL.dataset_size), num_batches=$(DL.num_batches), batch_size=$(DL.batch_size), shuffle=$(DL.shuffle), drop_last=$(DL.drop_last))")

@doc raw"""
    reshuffle!(data_loader::DataLoader)

Manually shuffle the data loader (even if shuffle is disabled in the given data loader).
It is recommended to reshuffle after each epoch during training.
"""
function reshuffle!(data_loader::DataLoader)
    data_loader.indices_batches = load_indices_batches(data_loader.dataset_size, 
                                    batch_size=data_loader.batch_size,
                                    shuffle=true,
                                    drop_last=data_loader.drop_last)
end

end # end of module "GradValley"