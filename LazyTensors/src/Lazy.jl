module Lazy

# Struct allowing for lazy evaluation of operations on AbstractArrays
# A LazyElementwiseOperation is defined by two same-sized AbstractArrays
# together with an operation. The operations are carried out when the
# LazyElementwiseOperation is indexed.
struct LazyElementwiseOperation{T,D,Op, T1<:AbstractArray{T,D}, T2 <: AbstractArray{T,D}} <: AbstractArray{T,D}
    a::T1
    b::T2

    function LazyElementwiseOperation{T,D,Op}(a::T1,b::T2) where {T,D,Op, T1<:AbstractArray{T,D}, T2<:AbstractArray{T,D}}
        #TODO: Remove assert? Asserts are not removed when compiling with
        # optimization flags. If so, need to handle boundschecking proparly.
        @assert size(a) == size(b)
        return new{T,D,Op,T1,T2}(a,b)
    end
end

Base.size(v::LazyElementwiseOperation) = size(v.a)

# NOTE: Boundschecking in getindex functions now assumes that the size of the
# vectors in the LazyElementwiseOperation are the same size. If we remove the
# size assertion in the constructor we might have to handle
# boundschecking differently.
Base.@propagate_inbounds @inline function Base.getindex(leo::LazyElementwiseOperation{T,D,:+}, I...) where {T,D}
    @boundscheck if !checkbounds(Bool,leo.a,I...)
        throw(BoundsError([leo],[I...]))
    end
    return leo.a[I...] + leo.b[I...]
end
Base.@propagate_inbounds @inline function Base.getindex(leo::LazyElementwiseOperation{T,D,:-}, I...) where {T,D}
    @boundscheck if !checkbounds(Bool,leo.a,I...)
        throw(BoundsError([leo],[I...]))
    end
    return leo.a[I...] - leo.b[I...]
end
Base.@propagate_inbounds @inline function Base.getindex(leo::LazyElementwiseOperation{T,D,:*}, I...) where {T,D}
    @boundscheck if !checkbounds(Bool,leo.a,I...)
        throw(BoundsError([leo],[I...]))
    end
    return leo.a[I...] * leo.b[I...]
end
Base.@propagate_inbounds @inline function Base.getindex(leo::LazyElementwiseOperation{T,D,:/}, I...) where {T,D}
    @boundscheck if !checkbounds(Bool,leo.a,I...)
        throw(BoundsError([leo],[I...]))
    end
    return leo.a[I...] / leo.b[I...]
end

# Define lazy operations for AbstractArrays. Operations constructs a LazyElementwiseOperation which
# can later be indexed into. Lazy operations are denoted by the usual operator followed by a tilde
@inline +̃(a::AbstractArray{T,D},b::AbstractArray{T,D}) where {T,D} = LazyElementwiseOperation{T,D,:+}(a,b)
@inline -̃(a::AbstractArray{T,D},b::AbstractArray{T,D}) where {T,D} = LazyElementwiseOperation{T,D,:-}(a,b)
@inline *̃(a::AbstractArray{T,D},b::AbstractArray{T,D}) where {T,D} = LazyElementwiseOperation{T,D,:*}(a,b)
@inline /̃(a::AbstractArray{T,D},b::AbstractArray{T,D}) where {T,D} = LazyElementwiseOperation{T,D,:/}(a,b)

# Abstract type for which the normal operations are defined by their
# lazy counterparts
abstract type LazyArray{T,D} <: AbstractArray{T,D} end;

Base.:+(a::LazyArray{T,D},b::AbstractArray{T,D}) where {T,D} = a +̃ b
Base.:+(a::AbstractArray{T,D}, b::LazyArray{T,D}) where {T,D} = b + a
Base.:-(a::LazyArray{T,D},b::AbstractArray{T,D}) where {T,D} = a -̃ b
Base.:-(a::AbstractArray{T,D}, b::LazyArray{T,D}) where {T,D} = a -̃ b
Base.:*(a::LazyArray{T,D},b::AbstractArray{T,D}) where {T,D} = a *̃ b
Base.:*(a::AbstractArray{T,D},b::LazyArray{T,D}) where {T,D} = b * a
# TODO: / seems to be ambiguous
# Base.:/(a::LazyArray{T,D},b::AbstractArray{T,D}) where {T,D} = a /̃ b
# Base.:/(a::AbstractArray{T,D},b::LazyArray{T,D}) where {T,D} = a /̃ b

export +̃, -̃, *̃, /̃, +, -, * #, /

end
