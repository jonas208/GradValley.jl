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
        batch = Array{eltype(array), num_dims}(undef, current_batch_size, size(array)...)
        push!(data_batch, batch)
    end
    data_batch = Tuple(data_batch)
    # fill the batches in the tuple with data
    for (i, index) in enumerate(indices_batch)
        element = get_function(index)
        for (j, batch) in enumerate(data_batch)
            ranges = [1:dim_size for dim_size in size(batch)[2:end]] # useful for indexing an array with arbitrary size
            batch[i, ranges...] = element[j]
        end
    end
    
    return data_batch
end

# is an iterable object for easily iterating over a dataset, each time an item is needed,
# it will be loaded just in time with the given get_function (instead of loading the entire data set directly -> more memory efficient)
# Important: The get_function has to take exactly one normal argument: the index of the item the function will return
# A tuple of arbitrary length is expected as return value, each element in this tuple must be an array. 
# The size and the type of the tuple or array is expected to be the same at each index.
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

# manually shuffle the data loader (even if shuffle is disabled in the given data loader)
# it is recommend to reshuffle after each epoch during training
function reshuffle!(data_loader::DataLoader)
    data_loader.indices_batches = load_indices_batches(data_loader.dataset_size, 
                                    batch_size=data_loader.batch_size,
                                    shuffle=true,
                                    drop_last=data_loader.drop_last)
end

end # end of module "GradValley"