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
export DomainSizeMismatch
export RangeSizeMismatch

include("lazy_tensor.jl")
include("tensor_types.jl")
include("lazy_array.jl")
include("lazy_tensor_operations.jl")
include("tuple_manipulation.jl")

# Applying lazy tensors to vectors
Base.:*(a::LazyTensor, v::AbstractArray) = LazyTensorApplication(a,v)
Base.:*(a::LazyTensor, b::LazyTensor) = throw(MethodError(Base.:*,(a,b)))
Base.:*(a::LazyTensor, args::Union{LazyTensor, AbstractArray}...) = foldr(*,(a,args...))

# Addition and subtraction of lazy tensors
Base.:+(s::LazyTensor, t::LazyTensor) = LazyTensorBinaryOperation{:+}(s,t)
Base.:-(s::LazyTensor, t::LazyTensor) = LazyTensorBinaryOperation{:-}(s,t)

# Composing lazy tensors
Base.:∘(s::LazyTensor, t::LazyTensor) = LazyTensorComposition(s,t)

# Outer products of tensors
⊗(a::LazyTensor, b::LazyTensor) = LazyOuterProduct(a,b)

end # module
