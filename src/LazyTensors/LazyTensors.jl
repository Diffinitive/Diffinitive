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
include("tuple_manipulation.jl")

# Applying lazy tensors to vectors
Base.:*(a::LazyTensor, v::AbstractArray) = LazyTensorApplication(a,v)
Base.:*(a::LazyTensor, b::LazyTensor) = throw(MethodError(Base.:*,(a,b)))
Base.:*(a::LazyTensor, args::Union{LazyTensor, AbstractArray}...) = foldr(*,(a,args...))

# Addition and subtraction of lazy tensors
Base.:+(tm1::LazyTensor{T,R,D}, tm2::LazyTensor{T,R,D}) where {T,R,D} = LazyTensorBinaryOperation{:+,T,R,D}(tm1,tm2)
Base.:-(tm1::LazyTensor{T,R,D}, tm2::LazyTensor{T,R,D}) where {T,R,D} = LazyTensorBinaryOperation{:-,T,R,D}(tm1,tm2)

# Composing lazy tensors
Base.:∘(s::LazyTensor, t::LazyTensor) = LazyTensorComposition(s,t)

# Outer products of tensors
⊗(a::LazyTensor, b::LazyTensor) = LazyOuterProduct(a,b)

end # module
