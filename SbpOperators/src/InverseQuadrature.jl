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

function LazyTensors.apply(Qi::InverseQuadrature{Dim,T}, v::AbstractArray{T,Dim}, I::NTuple{Dim,Index}) where {T,Dim}
    error("not implemented")
end

LazyTensors.apply_transpose(Qi::InverseQuadrature{Dim,T}, v::AbstractArray{T,2}, I::NTuple{2,Index}) where {Dim,T} = LazyTensors.apply(Q,v,I)

@inline function LazyTensors.apply(Qi::InverseQuadrature{1,T}, v::AbstractVector{T}, I::NTuple{1,Index}) where T
    @inbounds q = apply(Qi.Hi[1], v , I[1])
    return q
end

@inline function LazyTensors.apply(Qi::InverseQuadrature{2,T}, v::AbstractArray{T,2}, I::NTuple{2,Index}) where T
    # InverseQuadrature in x direction
    @inbounds vx = view(v, :, Int(I[2]))
    @inbounds qx_inv = apply(Qi.Hi[1], vx , I[1])
    # InverseQuadrature in y-direction
    @inbounds vy = view(v, Int(I[1]), :)
    @inbounds qy_inv = apply(Qi.Hi[2], vy, I[2])
    return qx_inv*qy_inv
end

"""
    InverseQuadrature{Dim,T<:Real,N,M,K} <: TensorMapping{T,Dim,Dim}

Implements the quadrature operator `Hi` of Dim dimension as a TensorMapping
"""
export InverseDiagonalNorm, closuresize
struct InverseDiagonalNorm{T<:Real,N,M} <: TensorOperator{T,1}
    h_inv::T # The reciprocl grid spacing could be included in the stencil already. Preferable?
    closure::NTuple{M,T}
    #TODO: Write a nice constructor
end

@inline function LazyTensors.apply(Hi::InverseDiagonalNorm{T}, v::AbstractVector{T}, I::NTuple{1,Index}) where T
    return @inbounds apply(Hi, v, I[1])
end

LazyTensors.apply_transpose(Hi::InverseQuadrature{Dim,T}, v::AbstractArray{T,2}, I::NTuple{2,Index}) where T = LazyTensors.apply(Hi,v,I)

@inline LazyTensors.apply(Hi::InverseDiagonalNorm, v::AbstractVector{T}, i::Index{Lower}) where T
    return @inbounds Hi.h_inv*Hi.closure[Int(i)]*v[Int(i)]
end
@inline LazyTensors.apply(Hi::InverseDiagonalNorm,v::AbstractVector{T}, i::Index{Upper}) where T
    N = length(v);
    return @inbounds Hi.h_inv*Hi.closure[N-Int(i)+1]v[Int(i)]
end

@inline LazyTensors.apply(Hi::InverseDiagonalNorm, v::AbstractVector{T}, i::Index{Interior}) where T
    return @inbounds Hi.h_inv*v[Int(i)]
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
