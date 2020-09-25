"""
    LazyArray{T,D} <: AbstractArray{T,D}

Array which is calcualted lazily when indexing.

A subtype of `LazyArray` will use lazy version of `+`, `-`, `*`, `/`.
"""
abstract type LazyArray{T,D} <: AbstractArray{T,D} end
export LazyArray

struct LazyConstantArray{T,D} <: LazyArray{T,D}
	val::T
	size::NTuple{D,Int}
end

Base.size(lca::LazyConstantArray) = lca.size
Base.getindex(lca::LazyConstantArray{T,D}, I::Vararg{Int,D}) where {T,D} = lca.val

"""
    LazyElementwiseOperation{T,D,Op} <: LazyArray{T,D}
Struct allowing for lazy evaluation of elementwise operations on AbstractArrays.

A LazyElementwiseOperation contains two arrays together with an operation.
The operations are carried out when the LazyElementwiseOperation is indexed.
"""
struct LazyElementwiseOperation{T,D,Op,T1<:AbstractArray{T,D},T2<:AbstractArray{T,D}} <: LazyArray{T,D}
    a::T1
    b::T2

    function LazyElementwiseOperation{T,D,Op}(a::T1,b::T2) where {T,D,Op,T1<:AbstractArray{T,D},T2<:AbstractArray{T,D}}
        @boundscheck if size(a) != size(b)
            throw(DimensionMismatch("dimensions must match"))
        end
        return new{T,D,Op,T1,T2}(a,b)
    end

end
LazyElementwiseOperation{T,D,Op}(a::AbstractArray{T,D},b::T) where {T,D,Op} = LazyElementwiseOperation{T,D,Op}(a, LazyConstantArray(b, size(a)))
LazyElementwiseOperation{T,D,Op}(a::T,b::AbstractArray{T,D}) where {T,D,Op} = LazyElementwiseOperation{T,D,Op}(LazyConstantArray(a,  size(b)), b)
# TODO: Move Op to be the first parameter? Compare to Binary operations

Base.size(v::LazyElementwiseOperation) = size(v.a)

evaluate(leo::LazyElementwiseOperation{T,D,:+}, I::Vararg{Int,D}) where {T,D} = leo.a[I...] + leo.b[I...]
evaluate(leo::LazyElementwiseOperation{T,D,:-}, I::Vararg{Int,D}) where {T,D} = leo.a[I...] - leo.b[I...]
evaluate(leo::LazyElementwiseOperation{T,D,:*}, I::Vararg{Int,D}) where {T,D} = leo.a[I...] * leo.b[I...]
evaluate(leo::LazyElementwiseOperation{T,D,:/}, I::Vararg{Int,D}) where {T,D} = leo.a[I...] / leo.b[I...]

# TODO: Make sure boundschecking is done properly and that the lenght of the vectors are equal
# NOTE: Boundschecking in getindex functions now assumes that the size of the
# vectors in the LazyElementwiseOperation are the same size. If we remove the
# size assertion in the constructor we might have to handle
# boundschecking differently.
Base.@propagate_inbounds @inline function Base.getindex(leo::LazyElementwiseOperation{T,D}, I::Vararg{Int,D}) where {T,D}
    @boundscheck if !checkbounds(Bool, leo.a, I...)
        throw(BoundsError([leo], I...))
    end
    return evaluate(leo, I...)
end

# Define lazy operations for AbstractArrays. Operations constructs a LazyElementwiseOperation which
# can later be indexed into. Lazy operations are denoted by the usual operator followed by a tilde
Base.@propagate_inbounds +̃(a::AbstractArray{T,D}, b::AbstractArray{T,D}) where {T,D} = LazyElementwiseOperation{T,D,:+}(a,b)
Base.@propagate_inbounds -̃(a::AbstractArray{T,D}, b::AbstractArray{T,D}) where {T,D} = LazyElementwiseOperation{T,D,:-}(a,b)
Base.@propagate_inbounds *̃(a::AbstractArray{T,D}, b::AbstractArray{T,D}) where {T,D} = LazyElementwiseOperation{T,D,:*}(a,b)
Base.@propagate_inbounds /̃(a::AbstractArray{T,D}, b::AbstractArray{T,D}) where {T,D} = LazyElementwiseOperation{T,D,:/}(a,b)

Base.@propagate_inbounds +̃(a::AbstractArray{T,D}, b::T) where {T,D} = LazyElementwiseOperation{T,D,:+}(a,b)
Base.@propagate_inbounds -̃(a::AbstractArray{T,D}, b::T) where {T,D} = LazyElementwiseOperation{T,D,:-}(a,b)
Base.@propagate_inbounds *̃(a::AbstractArray{T,D}, b::T) where {T,D} = LazyElementwiseOperation{T,D,:*}(a,b)
Base.@propagate_inbounds /̃(a::AbstractArray{T,D}, b::T) where {T,D} = LazyElementwiseOperation{T,D,:/}(a,b)

Base.@propagate_inbounds +̃(a::T, b::AbstractArray{T,D}) where {T,D} = LazyElementwiseOperation{T,D,:+}(a,b)
Base.@propagate_inbounds -̃(a::T, b::AbstractArray{T,D}) where {T,D} = LazyElementwiseOperation{T,D,:-}(a,b)
Base.@propagate_inbounds *̃(a::T, b::AbstractArray{T,D}) where {T,D} = LazyElementwiseOperation{T,D,:*}(a,b)
Base.@propagate_inbounds /̃(a::T, b::AbstractArray{T,D}) where {T,D} = LazyElementwiseOperation{T,D,:/}(a,b)



# NOTE: Är det knas att vi har till exempel * istället för .* ??
# Oklart om det ens går att lösa..
Base.@propagate_inbounds Base.:+(a::LazyArray{T,D}, b::LazyArray{T,D}) where {T,D} = a +̃ b
Base.@propagate_inbounds Base.:+(a::LazyArray{T,D}, b::AbstractArray{T,D}) where {T,D} = a +̃ b
Base.@propagate_inbounds Base.:+(a::AbstractArray{T,D}, b::LazyArray{T,D}) where {T,D} = a +̃ b

Base.@propagate_inbounds Base.:-(a::LazyArray{T,D}, b::LazyArray{T,D}) where {T,D} = a -̃ b
Base.@propagate_inbounds Base.:-(a::LazyArray{T,D}, b::AbstractArray{T,D}) where {T,D} = a -̃ b
Base.@propagate_inbounds Base.:-(a::AbstractArray{T,D}, b::LazyArray{T,D}) where {T,D} = a -̃ b

# Element wise operation for `*` and `\` are not overloaded due to conflicts with the behavior
# of regular `*` and `/` for AbstractArrays. Use tilde versions instead.

export +̃, -̃, *̃, /̃
