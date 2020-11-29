"""
diagonal_quadrature(g,quadrature_closure)

Constructs the diagonal quadrature operator `H` on a grid of `Dim` dimensions as
a `TensorMapping`. The one-dimensional operator is a `DiagonalQuadrature`, while
the multi-dimensional operator is the outer-product of the
one-dimensional operators in each coordinate direction.
"""
function diagonal_quadrature(g::EquidistantGrid{Dim}, quadrature_closure) where Dim
    H = DiagonalQuadrature(restrict(g,1), quadrature_closure)
    for i ∈ 2:Dim
        H = H⊗DiagonalQuadrature(restrict(g,i), quadrature_closure)
    end
    return H
end
export diagonal_quadrature

"""
    DiagonalQuadrature{T,M} <: TensorMapping{T,1,1}

Implements the one-dimensional diagonal quadrature operator as a `TensorMapping`
The quadrature is defined by the quadrature interval length `h`, the quadrature
closure weights `closure` and the number of quadrature intervals `size`. The
interior stencil has the weight 1.
"""
struct DiagonalQuadrature{T,M} <: TensorMapping{T,1,1}
    h::T
    closure::NTuple{M,T}
    size::Tuple{Int}
end
export DiagonalQuadrature

"""
    DiagonalQuadrature(g, quadrature_closure)

Constructs the `DiagonalQuadrature` `H` on the `EquidistantGrid` `g` with
`H.closure` specified by  `quadrature_closure`.
"""
function DiagonalQuadrature(g::EquidistantGrid{1}, quadrature_closure)
    return DiagonalQuadrature(spacing(g)[1], quadrature_closure, size(g))
end

LazyTensors.range_size(H::DiagonalQuadrature) = H.size
LazyTensors.domain_size(H::DiagonalQuadrature) = H.size

"""
    apply(H::DiagonalQuadrature{T}, v::AbstractVector{T}, i) where T
Implements the application `(H*v)[i]` an `Index{R}` where `R` is one of the regions
`Lower`,`Interior`,`Upper`.
"""
function LazyTensors.apply(H::DiagonalQuadrature{T}, v::AbstractVector{T}, i::Index{Lower}) where T
    return @inbounds H.h*H.closure[Int(i)]*v[Int(i)]
end

function LazyTensors.apply(H::DiagonalQuadrature{T},v::AbstractVector{T}, i::Index{Upper}) where T
    N = length(v);
    return @inbounds H.h*H.closure[N-Int(i)+1]*v[Int(i)]
end

function LazyTensors.apply(H::DiagonalQuadrature{T}, v::AbstractVector{T}, i::Index{Interior}) where T
    return @inbounds H.h*v[Int(i)]
end

function LazyTensors.apply(H::DiagonalQuadrature{T},  v::AbstractVector{T}, i) where T
    N = length(v);
    r = getregion(i, closure_size(H), N)

    return LazyTensors.apply(H, v, Index(i, r))
end

"""
    apply(H::DiagonalQuadrature{T}, v::AbstractVector{T}, I::Index) where T
Implements the application (H'*v)[I]. The operator is self-adjoint.
"""
LazyTensors.apply_transpose(H::DiagonalQuadrature{T}, v::AbstractVector{T}, i) where T = LazyTensors.apply(H,v,i)

"""
    closure_size(H)
Returns the size of the closure stencil of a DiagonalQuadrature `H`.
"""
closure_size(H::DiagonalQuadrature{T,M}) where {T,M} = M
