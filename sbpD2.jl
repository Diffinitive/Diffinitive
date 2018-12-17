
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