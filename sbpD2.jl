
struct D2{T}
    quadratureClosure::Vector{T}
    innerStencil::Vector{T}
    closureStencils::Matrix{T}
    eClosure::Vector{T}
    dClosure::Vector{T}
end
