module LazyTensors

export LazyTensor
export apply
export range_dim, domain_dim
export range_size, domain_size

export TensorApplication
export TensorComposition
export TensorNegation
export TensorSum
export outer_product
export IdentityTensor
export ZeroTensor
export ScalingTensor
export DiagonalTensor
export DenseTensor
export InflatedTensor
export ⊗
export DomainSizeMismatch
export RangeSizeMismatch
export componentview
export ArrayComponentView

export LazyArray
export LazyFunctionArray
export +̃, -̃, *̃, /̃

include("lazy_tensor.jl")
include("tensor_types.jl")
include("lazy_array.jl")
include("lazy_tensor_operations.jl")
include("lazy_tensor_operators.jl")
include("tuple_manipulation.jl")
include("componentview.jl")

end # module
