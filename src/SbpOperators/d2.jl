export D2, closuresize

@enum Parity begin
    odd = -1
    even = 1
end


# TBD: Can this be deleted when this branch is finished?
struct D2{T,M}
    innerStencil::Stencil{T}
    closureStencils::NTuple{M,Stencil{T}}
    eClosure::Stencil{T}
    dClosure::Stencil{T}
    quadratureClosure::NTuple{M,Stencil{T}}
    parity::Parity
end

closuresize(D::D2{T,M}) where {T,M} = M
