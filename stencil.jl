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
    # TBD: Rearrange to mark with @boundscheck?
    if s.range[1] <= i <= s.range[2]
        return s.weights[1 + i - s.range[1]]
    else
        return 0
    end
end

function apply(s::Stencil, v::AbstractVector, i::Int)
    w = zero(eltype(v))
    for j ∈ s.range[1]:s.range[2]
        w += s[j]*v[i+j] # TBD: Make this without boundschecks?
    end
    return w
end
