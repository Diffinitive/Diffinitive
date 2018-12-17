
struct D2{T}
    quadratureClosure::Vector{T}
    innerStencil::Vector{T}
    closureStencils::Matrix{T}
    eClosure::Vector{T}
    dClosure::Vector{T}
end

function closureSize(D::D2)::Int
    return length(quadratureClosure)
end
