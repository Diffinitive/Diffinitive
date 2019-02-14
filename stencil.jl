struct Stencil{T<:Real,N}
    range::Tuple{Int,Int}
    weights::NTuple{N,T}
end

function flip(s::Stencil)
    range = (-s.range[2], -s.range[1])
    return Stencil(range, reverse(s.weights))
end

# Provides index into the Stencil based on offset for the root element
@inline function Base.getindex(s::Stencil, i::Int)
    @boundscheck if i < s.range[1] || s.range[2] < i
        return eltype(s.weights)(0)
    end
    return s.weights[1 + i - s.range[1]]
end

Base.@propagate_inbounds @inline function apply(s::Stencil{T,N}, v::AbstractVector, i::Int) where {T,N}
    w = s.weights[1]*v[i+ s.range[1]]
    @simd for k ∈ 2:N
        w += s.weights[k]*v[i+ s.range[1] + k-1]
    end
    return w
end

# TODO: Fix loop unrolling here as well. Then we can also remove Base.getindex(::Stencil)
Base.@propagate_inbounds @inline function apply_backwards(s::Stencil, v::AbstractVector, i::Int)
    w = zero(eltype(v))
    for j ∈ s.range[2]:-1:s.range[1]
        @inbounds weight = s[j]
        w += weight*v[i-j]
    end
    return w
end
