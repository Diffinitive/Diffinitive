struct Stencil{T<:Real,N}
    range::Tuple{Int,Int}
    weights::NTuple{N,T}

    function Stencil(range::Tuple{Int,Int},weights::NTuple{N,T}) where {T <: Real, N}
        @assert range[2]-range[1]+1 == N
        new{T,N}(range,weights)
    end
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
    w = s.weights[1]*v[i + s.range[1]]
    @simd for k ∈ 2:N
        w += s.weights[k]*v[i + s.range[1] + k-1]
    end
    return w
end

Base.@propagate_inbounds @inline function apply_backwards(s::Stencil{T,N}, v::AbstractVector, i::Int) where {T,N}
    w = s.weights[N]*v[i - s.range[2]]
    @simd for k ∈ N-1:-1:1
        w += s.weights[k]*v[i - s.range[1] - k + 1]
    end
    return w
end
