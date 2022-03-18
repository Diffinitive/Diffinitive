module LazyTensors

export LazyTensorApplication
export LazyTensorTranspose
export LazyTensorComposition
export LazyLinearMap
export IdentityTensor
export ScalingTensor
export InflatedLazyTensor
export LazyOuterProduct
export ⊗
export SizeMismatch

include("lazy_tensor.jl")
include("lazy_array.jl")
include("lazy_tensor_operations.jl")

end # module
