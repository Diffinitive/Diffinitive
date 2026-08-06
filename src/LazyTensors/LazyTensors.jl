module LazyTensors

using StaticArrays

export LazyTensor
export apply
export range_dim, domain_dim
export range_size, domain_size

export TensorApplication
export TensorComposition
export TensorNegation
export TensorSum
export TensorOuterProduct
export IdentityTensor
export ZeroTensor
export ScalingTensor
export DiagonalTensor
export DenseTensor
export InflatedTensor
export VectorTensor
export VectorDotTensor
export MatrixTensor
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
    *(t::LazyTensor, v::AbstractArray)

Lazy application of a `LazyTensor` to an array. The elements of the result are
computed on indexing the returned array.

See also [`TensorApplication`](@ref)
"""
Base.:*(t::LazyTensor, v::AbstractArray) = TensorApplication(t, v)
Base.:*(t1::LazyTensor, t2::LazyTensor) = throw(MethodError(Base.:*, (t1, t2)))
Base.:*(t::LazyTensor, args::Union{LazyTensor, AbstractArray}...) = foldr(*, (t, args...))
# TODO: Simplification of application with identity?
#       Base.:*(t::IdentityTensor, v::AbstractArray) = v
# Would this lead to type-unstable code?


"""
    *(a, t::LazyTensor)
    *(t::LazyTensor, a)

Lazy multiplication of a `LazyTensor` and a constant, resulting in a scaled `LazyTensor`.

See also: [`TensorComposition`](@ref), [`ScalingTensor`](@ref).
"""
Base.:*(a, t::LazyTensor) = TensorComposition(ScalingTensor(a, range_size(t)), t)
Base.:*(t::LazyTensor, a) = a*t
#TODO: Should we not have e.g. a::Number, for clearer error handeling?
#      Only case I can see where we would want something else if we let `a` be some
#      heap-allocated scalar, e.g. a = @view b[1] for some array b. But then it is 
#      not really a constant so I think we don't need to support this case here.

"""
    +(t1::LazyTensor, t2::LazyTensor)

Lazy summation of two `LazyTensor`s. Provides basic simplifications when
adding `TensorSum`s, and `ZeroTensor`s.

See also: [`TensorSum`](@ref).
"""
Base.:+(t1::LazyTensor, t2::LazyTensor) = TensorSum(t1, t2)
#  Addition and subtraction of lazy tensors
Base.:+(ts::LazyTensor...) = foldl(+, ts) # Break the multi argument + into regular binary + to allow pariwise specialisations to work.
## Specializations to flatten the nesting of tensors. This helps Julia during inference.
# TODO: Move these to TensorSum instead of Base.:+? Seems like we will always want this
Base.:+(t1::TensorSum, t2::TensorSum) = TensorSum(t1.ts..., t2.ts...)
Base.:+(t1::TensorSum, t2::LazyTensor) = TensorSum(t1.ts..., t2)
Base.:+(t1::LazyTensor, t2::TensorSum) = TensorSum(t1, t2.ts...)
## Addition of zero
Base.:+(t1::LazyTensor, t2::ZeroTensor) = (check_equal_size(t1, t2); t1)
Base.:+(t1::ZeroTensor, t2::LazyTensor) = (check_equal_size(t1, t2); t2)
Base.:+(t1::ZeroTensor, t2::ZeroTensor) = (check_equal_size(t1, t2); t1) # Resolve ambiguity
Base.:+(t1::TensorSum, t2::ZeroTensor) = (check_equal_size(t1, t2); t1) # Resolve ambiguity
Base.:+(t1::ZeroTensor, t2::TensorSum) = (check_equal_size(t1, t2); t2) # Resolve ambiguity


"""
    -(t::LazyTensor)

Negation of a `LazyTensor`. Provides simplifications when
negating `ZeroTensor`s.

See also: [`TensorNegation`](@ref).
"""
Base.:-(t::LazyTensor) = TensorNegation(t)
Base.:-(t::ZeroTensor) = t


"""
    -(t1::LazyTensor, t2::LazyTensor)

Lazy subtraction of `LazyTensor`s. Provides the same simplifications as the
summation of `LazyTensor`s.
"""
Base.:-(t1::LazyTensor, t2::LazyTensor) = t1 + (-t2)


"""
    ∘(t1::LazyTensor, t2::LazyTensor)

Lazy composition of `LazyTensor`s. Provides basic simplifications when
composing with `ZeroTensor`s and `IdentityTensor`s.

See also: [`TensorComposition`](@ref).
"""
Base.:∘(t1::LazyTensor, t2::LazyTensor) = TensorComposition(t1, t2)
Base.:∘(tcomp::TensorComposition, t::LazyTensor) = tcomp.t1∘(tcomp.t2∘t) # TODO: Move TensorComposition. Seems like we will always want this
## Composing with identity
Base.:∘(t1::LazyTensor, t2::IdentityTensor) = (check_composable(t1, t2); t1)
Base.:∘(t1::IdentityTensor, t2::LazyTensor) = (check_composable(t1, t2); t2)
Base.:∘(t1::IdentityTensor, t2::IdentityTensor) = (check_composable(t1, t2); t1) # Resolve ambiguity
Base.:∘(t1::TensorComposition, t2::IdentityTensor) = (check_composable(t1, t2); t1) # Resolve ambiguity
## Composing with zero
Base.:∘(t1::LazyTensor, t2::ZeroTensor) = (check_composable(t1, t2); ZeroTensor(range_size(t1), domain_size(t2)))
Base.:∘(t1::ZeroTensor, t2::LazyTensor) = (check_composable(t1, t2); ZeroTensor(range_size(t1), domain_size(t2)))
Base.:∘(t1::ZeroTensor, t2::ZeroTensor) = (check_composable(t1, t2); ZeroTensor(range_size(t1), domain_size(t2))) # Resolve ambiguity
Base.:∘(t1::IdentityTensor, t2::ZeroTensor) = (check_composable(t1, t2); ZeroTensor(range_size(t1), domain_size(t2))) # Resolve ambiguity
Base.:∘(t1::ZeroTensor, t2::IdentityTensor) = (check_composable(t1, t2); ZeroTensor(range_size(t1), domain_size(t2))) # Resolve ambiguity
Base.:∘(t1::TensorComposition, t2::ZeroTensor) = (check_composable(t1, t2); ZeroTensor(range_size(t1), domain_size(t2))) # Resolve ambiguity


"""
    ⊗(t1::LazyTensor, t2::LazyTensor)

Lazy outer product of `LazyTensor`s. Provides basic simplifications when
forming outer products with `ZeroTensor`s and `IdentityTensor`s.

See also: [`TensorOuterProduct`](@ref).
"""
⊗(t1::LazyTensor, t2::LazyTensor) = TensorOuterProduct(t1, t2)
# Outer products with IdentityTensor
⊗(t1::IdentityTensor, t2::IdentityTensor) = IdentityTensor(t1.size..., t2.size...)
⊗(t1::LazyTensor, t2::IdentityTensor) = InflatedTensor(t1, t2)
⊗(t1::IdentityTensor, t2::LazyTensor) = InflatedTensor(t1, t2)
# Outer product with ZeroTensor
⊗(t1::LazyTensor, t2::ZeroTensor) = ZeroTensor((range_size(t1)..., range_size(t2)...), (domain_size(t1)..., domain_size(t2)...))
⊗(t1::ZeroTensor, t2::LazyTensor) = ZeroTensor((range_size(t1)..., range_size(t2)...), (domain_size(t1)..., domain_size(t2)...))
⊗(t1::ZeroTensor, t2::ZeroTensor) = ZeroTensor((range_size(t1)..., range_size(t2)...), (domain_size(t1)..., domain_size(t2)...)) # Resolve ambiguity
⊗(t1::ZeroTensor, t2::IdentityTensor) = ZeroTensor((range_size(t1)..., range_size(t2)...), (domain_size(t1)..., domain_size(t2)...)) # Resolve ambiguity
⊗(t1::IdentityTensor, t2::ZeroTensor) = ZeroTensor((range_size(t1)..., range_size(t2)...), (domain_size(t1)..., domain_size(t2)...)) # Resolve ambiguity

end # module
