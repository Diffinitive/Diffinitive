module LazyTensors

export LazyTensorMappingApplication
export LazyTensorMappingTranspose
export TensorMappingComposition
export LazyLinearMap
export IdentityMapping
export ScalingTensor
export InflatedTensorMapping
export LazyOuterProduct
export ⊗
export SizeMismatch

include("tensor_mapping.jl")
include("lazy_array.jl")
include("lazy_tensor_operations.jl")

end # module
