# At the moment the grid property is used all over. It could possibly be removed if we implement all the 1D operators as TensorMappings
"""
    Quadrature{Dim,T<:Real,N,M,K} <: TensorMapping{T,Dim,Dim}

Implements the quadrature operator `Q` of Dim dimension as a TensorMapping
The multi-dimensional tensor operator consists of a tuple of 1D DiagonalQuadrature
tensor operators.
"""
struct Quadrature{Dim,T<:Real,N,M} <: TensorOperator{T,Dim}
    H::NTuple{Dim,DiagonalQuadrature{T,N,M}}
end
export Quadrature

LazyTensors.domain_size(Q::Quadrature{Dim}, range_size::NTuple{Dim,Integer}) where Dim = range_size

function LazyTensors.apply(Q::Quadrature{Dim,T}, v::AbstractArray{T,Dim}, I::NTuple{Dim,Index}) where {T,Dim}
    error("not implemented")
end

LazyTensors.apply_transpose(Q::Quadrature{Dim,T}, v::AbstractArray{T,2}, I::NTuple{2,Index}) where {Dim,T} = LazyTensors.apply(Q,v,I)

@inline function LazyTensors.apply(Q::Quadrature{1,T}, v::AbstractVector{T}, I::NTuple{1,Index}) where T
    @inbounds q = apply(Q.H[1], v , I[1])
    return q
end

@inline function LazyTensors.apply(Q::Quadrature{2,T}, v::AbstractArray{T,2}, I::NTuple{2,Index}) where T
    # Quadrature in x direction
    @inbounds vx = view(v, :, Int(I[2]))
    @inbounds qx = apply(Q.H[1], vx , I[1])
    # Quadrature in y-direction
    @inbounds vy = view(v, Int(I[1]), :)
    @inbounds qy = apply(Q.H[2], vy, I[2])
    return qx*qy
end

"""
    Quadrature{Dim,T<:Real,N,M,K} <: TensorMapping{T,Dim,Dim}

Implements the quadrature operator `H` of Dim dimension as a TensorMapping
"""
struct DiagonalQuadrature{T<:Real,N,M} <: TensorOperator{T,1}
    h::T # The grid spacing could be included in the stencil already. Preferable?
    closure::NTuple{M,T}
    #TODO: Write a nice constructor
end

@inline function LazyTensors.apply(H::DiagonalQuadrature{T}, v::AbstractVector{T}, I::NTuple{1,Index}) where T
    return @inbounds apply(H, v, I[1])
end

LazyTensors.apply_transpose(H::Quadrature{Dim,T}, v::AbstractArray{T,2}, I::NTuple{2,Index}) where T = LazyTensors.apply(H,v,I)

@inline LazyTensors.apply(H::DiagonalQuadrature, v::AbstractVector{T}, i::Index{Lower}) where T
    return @inbounds H.h*H.closure[Int(i)]*v[Int(i)]
end
@inline LazyTensors.apply(H::DiagonalQuadrature,v::AbstractVector{T}, i::Index{Upper}) where T
    N = length(v);
    return @inbounds H.h*H.closure[N-Int(i)+1]v[Int(i)]
end

@inline LazyTensors.apply(H::DiagonalQuadrature, v::AbstractVector{T}, i::Index{Interior}) where T
    return @inbounds H.h*v[Int(i)]
end

function LazyTensors.apply(H::DiagonalQuadrature,  v::AbstractVector{T}, index::Index{Unknown}) where T
    N = length(v);
    r = getregion(Int(index), closuresize(H), N)
    i = Index(Int(index), r)
    return LazyTensors.apply(H, v, i)
end
export LazyTensors.apply

function closuresize(H::DiagonalQuadrature{T<:Real,N,M}) where {T,N,M}
    return M
end
