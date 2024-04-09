"""
    LazyArray{T,D} <: AbstractArray{T,D}

Array which is calculated lazily when indexing.

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
    LazyFunctionArray{F<:Function,T, D} <: LazyArray{T,D}

A lazy array where each element is defined by a function f(i,j,...)
"""
struct LazyFunctionArray{F<:Function,T, D} <: LazyArray{T,D}
    f::F
    size::NTuple{D,Int}
end
export LazyFunctionArray

function LazyFunctionArray(f::F, size::NTuple{D,Int}) where {F<:Function,D}
    T = typeof(f(ones(Int, D)...))
    return LazyFunctionArray{F,T,D}(f,size)
end

Base.size(lfa::LazyFunctionArray) = lfa.size

function Base.getindex(lfa::LazyFunctionArray{F,T,D}, I::Vararg{Int,D}) where {F,T,D}
    @boundscheck checkbounds(lfa, I...)
    return @inbounds @inline lfa.f(I...)
end


"""
    LazyElementwiseOperation{T,D,Op} <: LazyArray{T,D}
Struct allowing for lazy evaluation of element-wise operations on `AbstractArray`s.

A `LazyElementwiseOperation` contains two arrays together with an operation.
The operations are carried out when the `LazyElementwiseOperation` is indexed.
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

Base.size(v::LazyElementwiseOperation) = size(v.a)

Base.@propagate_inbounds evaluate(leo::LazyElementwiseOperation{T,D,:+}, I::Vararg{Int,D}) where {T,D} = leo.a[I...] + leo.b[I...]
Base.@propagate_inbounds evaluate(leo::LazyElementwiseOperation{T,D,:-}, I::Vararg{Int,D}) where {T,D} = leo.a[I...] - leo.b[I...]
Base.@propagate_inbounds evaluate(leo::LazyElementwiseOperation{T,D,:*}, I::Vararg{Int,D}) where {T,D} = leo.a[I...] * leo.b[I...]
Base.@propagate_inbounds evaluate(leo::LazyElementwiseOperation{T,D,:/}, I::Vararg{Int,D}) where {T,D} = leo.a[I...] / leo.b[I...]

function Base.getindex(leo::LazyElementwiseOperation{T,D}, I::Vararg{Int,D}) where {T,D}
    @boundscheck checkbounds(leo.a, I...)
    return @inbounds evaluate(leo, I...)
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



# Overload +,-,*,/ for LazyArrays 
# Element wise operation for `*` and `/` are not overloaded for due to conflicts with the behavior
# of regular `*` and `/` for AbstractArrays. Use tilde versions instead.
Base.@propagate_inbounds Base.:+(a::LazyArray{T,D}, b::LazyArray{T,D}) where {T,D} = a +̃ b
Base.@propagate_inbounds Base.:-(a::LazyArray{T,D}, b::LazyArray{T,D}) where {T,D} = a -̃ b

Base.@propagate_inbounds Base.:+(a::LazyArray{T,D}, b::AbstractArray{T,D}) where {T,D} = a +̃ b
Base.@propagate_inbounds Base.:-(a::LazyArray{T,D}, b::AbstractArray{T,D}) where {T,D} = a -̃ b

Base.@propagate_inbounds Base.:+(a::AbstractArray{T,D}, b::LazyArray{T,D}) where {T,D} = a +̃ b
Base.@propagate_inbounds Base.:-(a::AbstractArray{T,D}, b::LazyArray{T,D}) where {T,D} = a -̃ b

Base.@propagate_inbounds Base.:+(a::LazyArray{T,D}, b::T) where {T,D} = a +̃ b
Base.@propagate_inbounds Base.:-(a::LazyArray{T,D}, b::T) where {T,D} = a -̃ b

Base.@propagate_inbounds Base.:+(a::T, b::LazyArray{T,D}) where {T,D} = a +̃ b
Base.@propagate_inbounds Base.:-(a::T, b::LazyArray{T,D}) where {T,D} = a -̃  b

export +̃, -̃, *̃, /̃
