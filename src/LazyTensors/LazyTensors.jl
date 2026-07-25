module LazyTensors

export LazyTensor
export apply
export apply_transpose
export range_dim, domain_dim
export range_size, domain_size

export TensorApplication
export TensorTranspose
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
"""
    *(T::LazyTensor, v::AbstractArray)

Lazy application of a LazyTensor to an array. The elements of the result are
computed on indexing the returned array.
"""
Base.:*(a::LazyTensor, v::AbstractArray) = TensorApplication(a,v)
Base.:*(a::LazyTensor, b::LazyTensor) = throw(MethodError(Base.:*,(a,b)))
Base.:*(a::LazyTensor, args::Union{LazyTensor, AbstractArray}...) = foldr(*,(a,args...))
# TODO: Simplification of application with identity?

# Multiplication by constant
"""
    *(a, tm::LazyTensor)
    *(tm::LazyTensor, a)

Lazy multiplication of a lazy tensor and a constant, giving a resulting LazyTensor.
"""
Base.:*(a, tm::LazyTensor) = TensorComposition(ScalingTensor(a,range_size(tm)), tm)
Base.:*(tm::LazyTensor, a) = a*tm


#  Addition and subtraction of lazy tensors
Base.:+(ts::LazyTensor...) = foldl(+, ts) # Break the multi argument + into regular binary + to allow pariwise specialisations to work.
"""
    +(A::LazyTensor, B::LazyTensor)

Lazy summation of two `LazyTensor`s. Provides basic simplifications when
adding sums, and zero tensors.
"""
Base.:+(t::LazyTensor, s::LazyTensor) = TensorSum(t, s)
Base.:-(t::LazyTensor) = TensorNegation(t)
Base.:-(s::LazyTensor, t::LazyTensor) = s + (-t)
## Specializations to flatten the nesting of tensors. This helps Julia during inference.
Base.:+(t::TensorSum, s::TensorSum) = TensorSum(t.tms..., s.tms...)
Base.:+(t::TensorSum, s::LazyTensor) = TensorSum(t.tms..., s)
Base.:+(t::LazyTensor, s::TensorSum) = TensorSum(t, s.tms...)
## Addition of zero
Base.:+(t::LazyTensor, s::ZeroTensor) = (check_equal_size(t,s); t)
Base.:+(t::ZeroTensor, s::LazyTensor) = (check_equal_size(t,s); s)
Base.:+(t::ZeroTensor, s::ZeroTensor) = (check_equal_size(t,s); t) # Resolve ambiguity
Base.:+(t::TensorSum, s::ZeroTensor) = (check_equal_size(t,s); t) # Resolve ambiguity
Base.:+(t::ZeroTensor, s::TensorSum) = (check_equal_size(t,s); s) # Resolve ambiguity
Base.:-(t::ZeroTensor) = t

#TODO Write about the philosophy of operators and types in the docs. Operators
#for convenience. Types for full control.

# Composing lazy tensors
"""
    ∘(A::LazyTensor, B::LazyTensor)

Lazy composition of `LazyTensor`s. Provides basic simplifications when
composing with zero and identity tensors.
"""
Base.:∘(s::LazyTensor, t::LazyTensor) = TensorComposition(s,t)
Base.:∘(s::TensorComposition, t::LazyTensor) = s.t1∘(s.t2∘t)
## Composing with identity
Base.:∘(t::LazyTensor, s::IdentityTensor) = (check_composable(t,s); t)
Base.:∘(t::IdentityTensor, s::LazyTensor) = (check_composable(t,s); s)
Base.:∘(t::IdentityTensor, s::IdentityTensor) = (check_composable(t,s); t) # Resolve ambiguity
Base.:∘(t::TensorComposition, s::IdentityTensor) = (check_composable(t,s); t) # Resolve ambiguity
## Composing with zero
Base.:∘(t::LazyTensor, s::ZeroTensor) = (check_composable(t,s); ZeroTensor(range_size(t), domain_size(s)))
Base.:∘(t::ZeroTensor, s::LazyTensor) = (check_composable(t,s); ZeroTensor(range_size(t), domain_size(s)))
Base.:∘(t::ZeroTensor, s::ZeroTensor) = (check_composable(t,s); ZeroTensor(range_size(t), domain_size(s))) # Resolve ambiguity
Base.:∘(t::IdentityTensor, s::ZeroTensor) = (check_composable(t,s); ZeroTensor(range_size(t), domain_size(s))) # ResolveAmbiguity
Base.:∘(t::ZeroTensor, s::IdentityTensor) = (check_composable(t,s); ZeroTensor(range_size(t), domain_size(s))) # ResolveAmbiguity
Base.:∘(t::TensorComposition, s::ZeroTensor) = (check_composable(t,s); ZeroTensor(range_size(t), domain_size(s))) # ResolveAmbiguity

# Outer products of tensors
⊗(a::LazyTensor, b::LazyTensor) = LazyOuterProduct(a,b)

end # module
