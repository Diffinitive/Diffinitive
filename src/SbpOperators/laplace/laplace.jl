export Laplace
"""
    Laplace{Dim,T<:Real,N,M,K} <: TensorMapping{T,Dim,Dim}

Implements the Laplace operator `L` in Dim dimensions as a tensor operator
The multi-dimensional tensor operator consists of a tuple of 1D SecondDerivative
tensor operators.
"""
#export quadrature, inverse_quadrature, boundary_quadrature, boundary_value, normal_derivative
struct Laplace{Dim,T,N,M,K} <: TensorMapping{T,Dim,Dim}
    D2::NTuple{Dim,SecondDerivative{T,N,M,K}}
end

function Laplace(g::EquidistantGrid{Dim}, innerStencil, closureStencils) where Dim
    D2 = ()
    for i ∈ 1:Dim
        D2 = (D2..., SecondDerivative(subgrid(g,i), innerStencil, closureStencils))
    end

    return Laplace(D2)
end

LazyTensors.range_size(L::Laplace) = getindex.(range_size.(L.D2),1)
LazyTensors.domain_size(L::Laplace) = getindex.(domain_size.(L.D2),1)

function LazyTensors.apply(L::Laplace{Dim,T}, v::AbstractArray{T,Dim}, I::Vararg{Index,Dim}) where {T,Dim}
    error("not implemented")
end

# u = L*v
function LazyTensors.apply(L::Laplace{1,T}, v::AbstractVector{T}, I::Index) where T
    @inbounds u = LazyTensors.apply(L.D2[1],v,I)
    return u
end

function LazyTensors.apply(L::Laplace{2,T}, v::AbstractArray{T,2}, I::Index, J::Index) where T
    # 2nd x-derivative
    @inbounds vx = view(v, :, Int(J))
    @inbounds uᵢ = LazyTensors.apply(L.D2[1], vx , I)

    # 2nd y-derivative
    @inbounds vy = view(v, Int(I), :)
    @inbounds uᵢ += LazyTensors.apply(L.D2[2], vy , J)

    return uᵢ
end

# quadrature(L::Laplace) = Quadrature(L.op, L.grid)
# inverse_quadrature(L::Laplace) = InverseQuadrature(L.op, L.grid)
# boundary_value(L::Laplace, bId::CartesianBoundary) = BoundaryValue(L.op, L.grid, bId)
# normal_derivative(L::Laplace, bId::CartesianBoundary) = NormalDerivative(L.op, L.grid, bId)
# boundary_quadrature(L::Laplace, bId::CartesianBoundary) = BoundaryQuadrature(L.op, L.grid, bId)
# export NormalDerivative
# """
#     NormalDerivative{T,N,M,K} <: TensorMapping{T,2,1}
#
# Implements the boundary operator `d` as a TensorMapping
# """
# struct NormalDerivative{T,N,M,K} <: TensorMapping{T,2,1}
#     op::D2{T,N,M,K}
#     grid::EquidistantGrid{2}
#     bId::CartesianBoundary
# end
#
# # TODO: This is obviouly strange. Is domain_size just discarded? Is there a way to avoid storing grid in BoundaryValue?
# # Can we give special treatment to TensorMappings that go to a higher dim?
# function LazyTensors.range_size(e::NormalDerivative, domain_size::NTuple{1,Integer})
#     if dim(e.bId) == 1
#         return (UnknownDim, domain_size[1])
#     elseif dim(e.bId) == 2
#         return (domain_size[1], UnknownDim)
#     end
# end
# LazyTensors.domain_size(e::NormalDerivative, range_size::NTuple{2,Integer}) = (range_size[3-dim(e.bId)],)
#
# # TODO: Not type stable D:<
# # TODO: Make this independent of dimension
# function LazyTensors.apply(d::NormalDerivative{T}, v::AbstractArray{T}, I::NTuple{2,Index}) where T
#     i = I[dim(d.bId)]
#     j = I[3-dim(d.bId)]
#     N_i = size(d.grid)[dim(d.bId)]
#     h_inv = inverse_spacing(d.grid)[dim(d.bId)]
#     return apply_normal_derivative(d.op, h_inv, v[j], i, N_i, region(d.bId))
# end
#
# function LazyTensors.apply_transpose(d::NormalDerivative{T}, v::AbstractArray{T}, I::NTuple{1,Index}) where T
#     u = selectdim(v,3-dim(d.bId),Int(I[1]))
#     return apply_normal_derivative_transpose(d.op, inverse_spacing(d.grid)[dim(d.bId)], u, region(d.bId))
# end
#
# """
#     BoundaryQuadrature{T,N,M,K} <: TensorOperator{T,1}
#
# Implements the boundary operator `q` as a TensorOperator
# """
# export BoundaryQuadrature
# struct BoundaryQuadrature{T,N,M,K} <: TensorOperator{T,1}
#     op::D2{T,N,M,K}
#     grid::EquidistantGrid{2}
#     bId::CartesianBoundary
# end
#
#
# # TODO: Make this independent of dimension
# function LazyTensors.apply(q::BoundaryQuadrature{T}, v::AbstractArray{T,1}, I::NTuple{1,Index}) where T
#     h = spacing(q.grid)[3-dim(q.bId)]
#     N = size(v)
#     return apply_quadrature(q.op, h, v[I[1]], I[1], N[1])
# end
#
# LazyTensors.apply_transpose(q::BoundaryQuadrature{T}, v::AbstractArray{T,1}, I::NTuple{1,Index}) where T = LazyTensors.apply(q,v,I)
#
#
#
#
# struct Neumann{Bid<:BoundaryIdentifier} <: BoundaryCondition end
#
# function sat(L::Laplace{2,T}, bc::Neumann{Bid}, v::AbstractArray{T,2}, g::AbstractVector{T}, I::CartesianIndex{2}) where {T,Bid}
#     e = boundary_value(L, Bid())
#     d = normal_derivative(L, Bid())
#     Hᵧ = boundary_quadrature(L, Bid())
#     H⁻¹ = inverse_quadrature(L)
#     return (-H⁻¹*e*Hᵧ*(d'*v - g))[I]
# end
#
# struct Dirichlet{Bid<:BoundaryIdentifier} <: BoundaryCondition
#     tau::Float64
# end
#
# function sat(L::Laplace{2,T}, bc::Dirichlet{Bid}, v::AbstractArray{T,2}, g::AbstractVector{T}, i::CartesianIndex{2}) where {T,Bid}
#     e = boundary_value(L, Bid())
#     d = normal_derivative(L, Bid())
#     Hᵧ = boundary_quadrature(L, Bid())
#     H⁻¹ = inverse_quadrature(L)
#     return (-H⁻¹*(tau/h*e + d)*Hᵧ*(e'*v - g))[I]
#     # Need to handle scalar multiplication and addition of TensorMapping
# end

# function apply(s::MyWaveEq{D},  v::AbstractArray{T,D}, i::CartesianIndex{D}) where D
    #   return apply(s.L, v, i) +
# 		sat(s.L, Dirichlet{CartesianBoundary{1,Lower}}(s.tau),  v, s.g_w, i) +
# 		sat(s.L, Dirichlet{CartesianBoundary{1,Upper}}(s.tau),  v, s.g_e, i) +
# 		sat(s.L, Dirichlet{CartesianBoundary{2,Lower}}(s.tau),  v, s.g_s, i) +
# 		sat(s.L, Dirichlet{CartesianBoundary{2,Upper}}(s.tau),  v, s.g_n, i)
# end
