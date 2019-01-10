struct Stencil
    range::NTuple{2,Int}
    weights::Vector # TBD: Should this be a tuple?
    function Stencil(range, weights)
        width = range[2]-range[1]+1
        if width != length(weights)
            error("The width and the number of weights must be the same")
        end
        new(range, weights)
    end
end

function flip(s::Stencil)
    range = (-s.range[2], -s.range[1])
    s = Stencil(range, s.weights[end:-1:1])
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
    w = zero(eltype(v))
    for j ∈ s.range[1]:s.range[2]
        w += s[j]*v[i+j]
    end
    return w
end
