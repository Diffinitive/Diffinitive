"""
diagonal_quadrature(g,quadrature_closure)

Constructs the diagonal quadrature operator `H` on a grid of `Dim` dimensions as
a `TensorMapping`. The one-dimensional operator is a DiagonalQuadrature, while
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
    DiagonalQuadrature{Dim,T<:Real,N,M,K} <: TensorMapping{T,Dim,Dim}

Implements the diagonal quadrature operator `H` of Dim dimension as a TensorMapping
"""
struct DiagonalQuadrature{T,M} <: TensorMapping{T,1,1}
    h::T
    closure::NTuple{M,T}
    size::Tuple{Int}
end
export DiagonalQuadrature

function DiagonalQuadrature(g::EquidistantGrid{1}, quadrature_closure)
    return DiagonalQuadrature(spacing(g)[1], quadrature_closure, size(g))
end

LazyTensors.range_size(H::DiagonalQuadrature) = H.size
LazyTensors.domain_size(H::DiagonalQuadrature) = H.size

function LazyTensors.apply(H::DiagonalQuadrature{T}, v::AbstractVector{T}, I::Index) where T
    return @inbounds apply(H, v, I)
end

function LazyTensors.apply(H::DiagonalQuadrature{T}, v::AbstractVector{T}, I::Index{Lower}) where T
    return @inbounds H.h*H.closure[Int(I)]*v[Int(I)]
end

function LazyTensors.apply(H::DiagonalQuadrature{T},v::AbstractVector{T}, I::Index{Upper}) where T
    N = length(v);
    return @inbounds H.h*H.closure[N-Int(I)+1]*v[Int(I)]
end

function LazyTensors.apply(H::DiagonalQuadrature{T}, v::AbstractVector{T}, I::Index{Interior}) where T
    return @inbounds H.h*v[Int(I)]
end

function LazyTensors.apply(H::DiagonalQuadrature{T},  v::AbstractVector{T}, index::Index{Unknown}) where T
    N = length(v);
    r = getregion(Int(index), closuresize(H), N)
    i = Index(Int(index), r)
    return LazyTensors.apply(H, v, i)
end

LazyTensors.apply_transpose(H::DiagonalQuadrature{T}, v::AbstractVector{T}, I::Index) where T = LazyTensors.apply(H,v,I)

closuresize(H::DiagonalQuadrature{T,M}) where {T,M} = M
export closuresize
