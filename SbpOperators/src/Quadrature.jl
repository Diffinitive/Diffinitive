# At the moment the grid property is used all over. It could possibly be removed if we implement all the 1D operators as TensorMappings
"""
    Quadrature{Dim,T<:Real,N,M,K} <: TensorMapping{T,Dim,Dim}

Implements the quadrature operator `Q` of Dim dimension as a TensorMapping
The multi-dimensional tensor operator consists of a tuple of 1D DiagonalNorm H
tensor operators.
"""
export Quadrature
struct Quadrature{Dim,T<:Real,N,M} <: TensorOperator{T,Dim}
    H::NTuple{Dim,DiagonalNorm{T,N,M}}
end

LazyTensors.domain_size(Q::Quadrature{Dim}, range_size::NTuple{Dim,Integer}) where Dim = range_size

function LazyTensors.apply(Q::Quadrature{Dim,T}, v::AbstractArray{T,Dim}, I::Vararg{Index,Dim}) where {T,Dim}
    error("not implemented")
end

LazyTensors.apply_transpose(Q::Quadrature{Dim,T}, v::AbstractArray{T,Dim}, I::Vararg{Index,Dim}) where {Dim,T} = LazyTensors.apply(Q,v,I)

@inline function LazyTensors.apply(Q::Quadrature{1,T}, v::AbstractVector{T}, I::Index) where T
    @inbounds q = apply(Q.H[1], v , I)
    return q
end

@inline function LazyTensors.apply(Q::Quadrature{2,T}, v::AbstractArray{T,2}, I::Index, J::Index) where T
    # Quadrature in x direction
    @inbounds vx = view(v, :, Int(J))
    @inbounds qx = apply(Q.H[1], vx , I)
    # Quadrature in y-direction
    @inbounds vy = view(v, Int(I), :)
    @inbounds qy = apply(Q.H[2], vy, J)
    return qx*qy
end

"""
    DiagonalNorm{Dim,T<:Real,N,M,K} <: TensorMapping{T,Dim,Dim}

Implements the diagnoal norm operator `H` of Dim dimension as a TensorMapping
"""
export DiagonalNorm, closuresize, LazyTensors.apply
struct DiagonalNorm{T<:Real,N,M} <: TensorOperator{T,1}
    h::T # The grid spacing could be included in the stencil already. Preferable?
    closure::NTuple{M,T}
    #TODO: Write a nice constructor
end

@inline function LazyTensors.apply(H::DiagonalNorm{T}, v::AbstractVector{T}, I::Index) where T
    return @inbounds apply(H, v, I)
end

LazyTensors.apply_transpose(H::Quadrature{Dim,T}, v::AbstractArray{T,2}, I::Index) where T = LazyTensors.apply(H,v,I)

@inline LazyTensors.apply(H::DiagonalNorm, v::AbstractVector{T}, I::Index{Lower}) where T
    return @inbounds H.h*H.closure[Int(I)]*v[Int(I)]
end
@inline LazyTensors.apply(H::DiagonalNorm,v::AbstractVector{T}, I::Index{Upper}) where T
    N = length(v);
    return @inbounds H.h*H.closure[N-Int(I)+1]v[Int(I)]
end

@inline LazyTensors.apply(H::DiagonalNorm, v::AbstractVector{T}, I::Index{Interior}) where T
    return @inbounds H.h*v[Int(I)]
end

function LazyTensors.apply(H::DiagonalNorm,  v::AbstractVector{T}, index::Index{Unknown}) where T
    N = length(v);
    r = getregion(Int(index), closuresize(H), N)
    i = Index(Int(index), r)
    return LazyTensors.apply(H, v, i)
end

function closuresize(H::DiagonalNorm{T<:Real,N,M}) where {T,N,M}
    return M
end
