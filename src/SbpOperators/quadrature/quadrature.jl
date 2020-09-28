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

function LazyTensors.apply(Q::Quadrature{Dim,T}, v::AbstractArray{T,Dim}, I::Vararg{Index,Dim}) where {T,Dim}
    error("not implemented")
end

function LazyTensors.apply(Q::Quadrature{1,T}, v::AbstractVector{T}, I::Index) where T
    @inbounds q = apply(Q.H[1], v , I)
    return q
end

function LazyTensors.apply(Q::Quadrature{2,T}, v::AbstractArray{T,2}, I::Index, J::Index) where T
    # Quadrature in x direction
    @inbounds vx = view(v, :, Int(J))
    @inbounds qx = apply(Q.H[1], vx , I)
    # Quadrature in y-direction
    @inbounds vy = view(v, Int(I), :)
    @inbounds qy = apply(Q.H[2], vy, J)
    return qx*qy
end

LazyTensors.apply_transpose(Q::Quadrature{Dim,T}, v::AbstractArray{T,Dim}, I::Vararg{Index,Dim}) where {Dim,T} = LazyTensors.apply(Q,v,I...)
