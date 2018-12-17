
struct D2{T}
    quadratureClosure::Vector{T}
    innerStencil::Stencil
    closureStencils::Vector{Stencil} # TBD: Should this be a tuple?
    eClosure::Vector{T}
    dClosure::Vector{T}
end

function closureSize(D::D2)::Int
    return length(quadratureClosure)
end

struct Stencil{T}
    range::NTuple{2,Int}
    weights::Vector{T} # TBD: Should this be a tuple?
    function Stencil(range, weights)
        width = range[2]-range[1]+1
        if width != length(weights)
            error("The width and the number of weights must be the same")
        end
        new(range, weights)
    end
end

# Provides index into the Stencil based on offset for the root element
function Base.getindex(s::Stencil, i::Int)
    if s.range[1] <= i <= s.range[2]
        return s.weights[1 + i - s.range[1]]
    else
        return 0
    end
end

function apply(s::Stencil, v::AbstractVector, i::Int)
    w = zero(v[0])
    for j ∈ i+(s.range[1]:s.range[2])
        w += v[j]
    end
    return w
end
