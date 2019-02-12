struct Stencil{T<:Real,N}
    range::Tuple{Int,Int}
    weights::NTuple{N,T}
end

function flip(s::Stencil)
    range = (-s.range[2], -s.range[1])
    return Stencil(range, reverse(s.weights))
end

# Provides index into the Stencil based on offset for the root element
function Base.getindex(s::Stencil, i::Int)
    @boundscheck if i < s.range[1] || s.range[2] < i
        return eltype(s.weights)(0)
    end

    return s.weights[1 + i - s.range[1]]
end

Base.@propagate_inbounds function apply(s::Stencil, v::AbstractVector, i::Int)
    w = zero(eltype(v))
    for j ∈ s.range[1]:s.range[2]
        @inbounds weight = s[j]
        w += weight*v[i+j]
    end
    return w
end

Base.@propagate_inbounds function apply_backwards(s::Stencil, v::AbstractVector, i::Int)
    w = zero(eltype(v))
    for j ∈ s.range[2]:-1:s.range[1]
        @inbounds weight = s[j]
        w += weight*v[i-j]
    end
    return w
end
