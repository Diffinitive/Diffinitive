export InverseQuadrature
"""
    InverseQuadrature{Dim,T<:Real,M,K} <: TensorMapping{T,Dim,Dim}

Implements the inverse quadrature operator `Qi` of Dim dimension as a TensorMapping
The multi-dimensional tensor operator consists of a tuple of 1D InverseDiagonalInnerProduct
tensor operators.
"""
struct InverseQuadrature{Dim,T<:Real,M} <: TensorMapping{T,Dim,Dim}
    Hi::NTuple{Dim,InverseDiagonalInnerProduct{T,M}}
end


function InverseQuadrature(g::EquidistantGrid{Dim}, quadratureClosure) where Dim
    Hi = ()
    for i ∈ 1:Dim
        Hi = (Hi..., InverseDiagonalInnerProduct(restrict(g,i), quadratureClosure))
    end

    return InverseQuadrature(Hi)
end

LazyTensors.range_size(Hi::InverseQuadrature) = getindex.(range_size.(Hi.Hi),1)
LazyTensors.domain_size(Hi::InverseQuadrature) = getindex.(domain_size.(Hi.Hi),1)

LazyTensors.domain_size(Qi::InverseQuadrature{Dim}, range_size::NTuple{Dim,Integer}) where Dim = range_size

function LazyTensors.apply(Qi::InverseQuadrature{Dim,T}, v::AbstractArray{T,Dim}, I::Vararg{Index,Dim}) where {T,Dim}
    error("not implemented")
end

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

LazyTensors.apply_transpose(Qi::InverseQuadrature{Dim,T}, v::AbstractArray{T,Dim}, I::Vararg{Index,Dim}) where {Dim,T} = LazyTensors.apply(Qi,v,I...)
