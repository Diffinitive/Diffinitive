"""
    InverseQuadrature{Dim,T<:Real,N,M,K} <: TensorMapping{T,Dim,Dim}

Implements the inverse quadrature operator `Qi` of Dim dimension as a TensorOperator
The multi-dimensional tensor operator consists of a tuple of 1D InverseDiagonalNorm
tensor operators.
"""
export InverseQuadrature
struct InverseQuadrature{Dim,T<:Real,N,M} <: TensorOperator{T,Dim}
    Hi::NTuple{Dim,InverseDiagonalNorm{T,N,M}}
end

LazyTensors.domain_size(Qi::InverseQuadrature{Dim}, range_size::NTuple{Dim,Integer}) where Dim = range_size

function LazyTensors.apply(Qi::InverseQuadrature{Dim,T}, v::AbstractArray{T,Dim}, I::Vararg{Index,Dim}) where {T,Dim}
    error("not implemented")
end

LazyTensors.apply_transpose(Qi::InverseQuadrature{Dim,T}, v::AbstractArray{T,Dim}, I::Vararg{Index,Dim}) where {Dim,T} = LazyTensors.apply(Q,v,I)

@inline function LazyTensors.apply(Qi::InverseQuadrature{1,T}, v::AbstractVector{T}, I::Index) where T
    @inbounds q = apply(Qi.Hi[1], v , I)
    return q
end

@inline function LazyTensors.apply(Qi::InverseQuadrature{2,T}, v::AbstractArray{T,2}, I::Index, J::Index) where T
    # InverseQuadrature in x direction
    @inbounds vx = view(v, :, Int(J))
    @inbounds qx_inv = apply(Qi.Hi[1], vx , I)
    # InverseQuadrature in y-direction
    @inbounds vy = view(v, Int(I), :)
    @inbounds qy_inv = apply(Qi.Hi[2], vy, J)
    return qx_inv*qy_inv
end

"""
    DiagonalNorm{Dim,T<:Real,N,M,K} <: TensorMapping{T,Dim,Dim}

Implements the inverse diagnoal norm operator `Hi` of Dim dimension as a TensorMapping
"""
export InverseDiagonalNorm, closuresize
struct InverseDiagonalNorm{T<:Real,N,M} <: TensorOperator{T,1}
    h_inv::T # The reciprocl grid spacing could be included in the stencil already. Preferable?
    closure::NTuple{M,T}
    #TODO: Write a nice constructor
end

@inline function LazyTensors.apply(Hi::InverseDiagonalNorm{T}, v::AbstractVector{T}, I:Index) where T
    return @inbounds apply(Hi, v, I)
end

LazyTensors.apply_transpose(Hi::InverseQuadrature{Dim,T}, v::AbstractArray{T,2}, I::Index) where T = LazyTensors.apply(Hi,v,I)

@inline LazyTensors.apply(Hi::InverseDiagonalNorm, v::AbstractVector{T}, I::Index{Lower}) where T
    return @inbounds Hi.h_inv*Hi.closure[Int(i)]*v[Int(I)]
end
@inline LazyTensors.apply(Hi::InverseDiagonalNorm,v::AbstractVector{T}, I::Index{Upper}) where T
    N = length(v);
    return @inbounds Hi.h_inv*Hi.closure[N-Int(I)+1]v[Int(I)]
end

@inline LazyTensors.apply(Hi::InverseDiagonalNorm, v::AbstractVector{T}, I::Index{Interior}) where T
    return @inbounds Hi.h_inv*v[Int(I)]
end

function LazyTensors.apply(Hi::InverseDiagonalNorm,  v::AbstractVector{T}, index::Index{Unknown}) where T
    N = length(v);
    r = getregion(Int(index), closuresize(Hi), N)
    i = Index(Int(index), r)
    return LazyTensors.apply(Hi, v, i)
end
export LazyTensors.apply

function closuresize(Hi::InverseDiagonalNorm{T<:Real,N,M}) where {T,N,M}
    return M
end
