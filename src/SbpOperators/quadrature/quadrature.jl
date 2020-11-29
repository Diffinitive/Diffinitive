export Quadrature
"""
    Quadrature{Dim,T<:Real,N,M,K} <: TensorMapping{T,Dim,Dim}

Implements the quadrature operator `Q` of Dim dimension as a TensorMapping
The multi-dimensional tensor operator consists of a tuple of 1D DiagonalInnerProduct H
tensor operators.
"""
struct Quadrature{Dim,T<:Real,M} <: TensorMapping{T,Dim,Dim}
    H::NTuple{Dim,DiagonalInnerProduct{T,M}}
end

function Quadrature(g::EquidistantGrid{Dim}, quadratureClosure) where Dim
    H = ()
    for i ∈ 1:Dim
        H = (H..., DiagonalInnerProduct(restrict(g,i), quadratureClosure))
    end

    return Quadrature(H)
end

LazyTensors.range_size(H::Quadrature) = getindex.(range_size.(H.H),1)
LazyTensors.domain_size(H::Quadrature) = getindex.(domain_size.(H.H),1)

function LazyTensors.apply(Q::Quadrature{Dim,T}, v::AbstractArray{T,Dim}, I::Vararg{Any,Dim}) where {T,Dim}
    error("not implemented")
end

function LazyTensors.apply(Q::Quadrature{1,T}, v::AbstractVector{T}, i) where T
    @inbounds q = apply(Q.H[1], v , i)
    return q
end

function LazyTensors.apply(Q::Quadrature{2,T}, v::AbstractArray{T,2}, i, j) where T
    # Quadrature in x direction
    @inbounds vx = view(v, :, Int(j))
    @inbounds qx = apply(Q.H[1], vx , i)
    # Quadrature in y-direction
    @inbounds vy = view(v, Int(i), :)
    @inbounds qy = apply(Q.H[2], vy, j)
    return qx*qy
end

LazyTensors.apply_transpose(Q::Quadrature{Dim,T}, v::AbstractArray{T,Dim}, I::Vararg{Any,Dim}) where {Dim,T} = LazyTensors.apply(Q,v,I...)
