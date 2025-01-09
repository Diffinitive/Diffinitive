module LazyTensors

export LazyTensor
export apply
export apply_transpose
export range_dim, domain_dim
export range_size, domain_size

export TensorApplication
export TensorTranspose
export TensorComposition
export IdentityTensor
export ScalingTensor
export DiagonalTensor
export DenseTensor
export InflatedTensor
export LazyOuterProduct
export ⊗
export DomainSizeMismatch
export RangeSizeMismatch

export LazyArray
export LazyFunctionArray
export +̃, -̃, *̃, /̃

include("lazy_tensor.jl")
include("tensor_types.jl")
include("lazy_array.jl")
include("lazy_tensor_operations.jl")
include("tuple_manipulation.jl")

# Applying lazy tensors to vectors
Base.:*(a::LazyTensor, v::AbstractArray) = TensorApplication(a,v)
Base.:*(a::LazyTensor, b::LazyTensor) = throw(MethodError(Base.:*,(a,b)))
Base.:*(a::LazyTensor, args::Union{LazyTensor, AbstractArray}...) = foldr(*,(a,args...))

# Addition and subtraction of lazy tensors
Base.:+(ts::LazyTensor...) = ElementwiseTensorOperation{:+}(ts...)
Base.:-(s::LazyTensor, t::LazyTensor) = ElementwiseTensorOperation{:-}(s,t)

# Composing lazy tensors
Base.:∘(s::LazyTensor, t::LazyTensor) = TensorComposition(s,t)
Base.:∘(s::TensorComposition, t::LazyTensor) = s.t1∘(s.t2∘t)

# Outer products of tensors
⊗(a::LazyTensor, b::LazyTensor) = LazyOuterProduct(a,b)

end # module
