struct Stencil{T<:Real}
    range::NTuple{2,Int}
    weights::Vector{T} # Should this be a tuple?? (Check type stability)

    function Stencil(range, weights)
        width = range[2]-range[1]+1
        if width != length(weights)
            error("The width and the number of weights must be the same")
        end
        new{eltype(weights)}(range, weights)
    end
end

function flip(s::Stencil)
    range = (-s.range[2], -s.range[1])
    s = Stencil(range, s.weights[end:-1:1])
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
