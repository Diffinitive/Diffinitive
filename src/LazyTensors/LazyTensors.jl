module LazyTensors

export LazyTensor
export apply
export apply_transpose
export range_dim, domain_dim
export range_size, domain_size

export TensorApplication
export TensorComposition
export TensorNegation
export TensorSum
export IdentityTensor
export ZeroTensor
export ScalingTensor
export DiagonalTensor
export DenseTensor
export InflatedTensor
export LazyOuterProduct
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
include("tuple_manipulation.jl")
include("componentview.jl")

# Applying lazy tensors to vectors
Base.:*(a::LazyTensor, v::AbstractArray) = TensorApplication(a,v)
Base.:*(a::LazyTensor, b::LazyTensor) = throw(MethodError(Base.:*,(a,b)))
Base.:*(a::LazyTensor, args::Union{LazyTensor, AbstractArray}...) = foldr(*,(a,args...))

# Addition and subtraction of lazy tensors
Base.:+(ts::LazyTensor...) = TensorSum(ts...)
Base.:-(t::LazyTensor) = TensorNegation(t)
Base.:-(s::LazyTensor, t::LazyTensor) = s + (-t)
## Specializations to flatten the nesting of tensors. This helps Julia during inference.
Base.:+(t::TensorSum, s::TensorSum) = TensorSum(t.tms..., s.tms...)
Base.:+(t::TensorSum, s::LazyTensor) = TensorSum(t.tms..., s)
Base.:+(t::LazyTensor, s::TensorSum) = TensorSum(t, s.tms...)

# Composing lazy tensors
Base.:∘(s::LazyTensor, t::LazyTensor) = TensorComposition(s,t)
Base.:∘(s::TensorComposition, t::LazyTensor) = s.t1∘(s.t2∘t)

# Outer products of tensors
⊗(a::LazyTensor, b::LazyTensor) = LazyOuterProduct(a,b)

end # module
