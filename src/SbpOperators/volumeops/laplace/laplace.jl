"""
    Laplace(grid::EquidistantGrid{Dim}, inner_stencil, closure_stencils)

Creates the Laplace ooperator operator `Δ` as a `TensorMapping`

`Δ` approximates the Laplace operator ∑d²/xᵢ² , i = 1,...,`Dim` on `grid`, using
the stencil `inner_stencil` in the interior and a set of stencils `closure_stencils`
for the points in the closure regions.

On a one-dimensional `grid`, `Δ` is a `SecondDerivative`. On a multi-dimensional `grid`, `Δ` is the sum of
multi-dimensional `SecondDerivative`s where the sum is carried out lazily.
"""
function Laplace(grid::EquidistantGrid{Dim}, inner_stencil, closure_stencils) where Dim
    Δ = SecondDerivative(grid, inner_stencil, closure_stencils, 1)
    for d = 2:Dim
        Δ += SecondDerivative(grid, inner_stencil, closure_stencils, d)
    end
    return Δ
end
export Laplace

# quadrature(L::Laplace) = Quadrature(L.op, L.grid)
# inverse_quadrature(L::Laplace) = InverseQuadrature(L.op, L.grid)
# boundary_value(L::Laplace, bId::CartesianBoundary) = BoundaryValue(L.op, L.grid, bId)
# normal_derivative(L::Laplace, bId::CartesianBoundary) = NormalDerivative(L.op, L.grid, bId)
# boundary_quadrature(L::Laplace, bId::CartesianBoundary) = BoundaryQuadrature(L.op, L.grid, bId)

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
