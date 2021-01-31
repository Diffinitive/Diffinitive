"""
    Laplace{T,Dim,...}()
    Laplace(grid::EquidistantGrid, fn; order)

Implements the Laplace operator, approximating ∑d²/xᵢ² , i = 1,...,`Dim` as a
`TensorMapping`. Additionally, `Laplace` stores the quadrature, and boundary
operators relevant for constructing a SBP finite difference scheme as `TensorMapping`s.
"""
struct Laplace{T, Dim, Rb, TMdiffop<:TensorMapping{T,Dim,Dim}, # Differential operator tensor mapping
                           TMqop<:TensorMapping{T,Dim,Dim}, # Quadrature operator tensor mapping
                           TMbop<:TensorMapping{T,Rb,Dim}, # Boundary operator tensor mapping
                           TMbqop<:TensorMapping{T,Rb,Rb}, # Boundary quadrature tensor mapping
                           BID<:BoundaryIdentifier} <: TensorMapping{T,Dim,Dim}
    D::TMdiffop # Difference operator
    H::TMqop # Quadrature (norm) operator
    H_inv::TMqop # Inverse quadrature (norm) operator
    e::Dict{BID,TMbop} # Boundary restriction operators
    d::Dict{BID,TMbop} # Normal derivative operators
    H_boundary::Dict{BID,TMbqop} # Boundary quadrature operators
end
export Laplace

function Laplace(grid::EquidistantGrid, fn; order)
    # TODO: Removed once we can construct the volume and
    # boundary operators by op(grid, fn; order,...).
    # Read stencils
    op = read_D2_operator(fn; order)
    D_inner_stecil = op.innerStencil
    D_closure_stencils = op.closureStencils
    H_closure_stencils = op.quadratureClosure
    e_closure_stencil = op.eClosure
    d_closure_stencil = op.dClosure

    # Volume operators
    Δ =  laplace(grid, D_inner_stecil, D_closure_stencils)
    H =  DiagonalQuadrature(grid, H_closure_stencils)
    H⁻¹ =  InverseDiagonalQuadrature(grid, H_closure_stencils)

    # Pair boundary operators and boundary quadratures with the boundary ids
    e_pairs = ()
    d_pairs = ()
    Hᵧ_pairs = ()
    dims = collect(1:dimension(grid))
    for id ∈ boundary_identifiers(grid)
        # Boundary operators
        e_pairs = (e_pairs...,Pair(id,BoundaryRestriction(grid,e_closure_stencil,id)))
        d_pairs = (d_pairs...,Pair(id,NormalDerivative(grid,d_closure_stencil,id)))
        # Boundary quadratures
        # Construct these on the lower-dimensional grid in the
        # coordinite directions orthogonal to dim(id)
        orth_dims = dims[dims .!= dim(id)]
        orth_grid = restrict(grid,orth_dims)
        Hᵧ_pairs = (Hᵧ_pairs...,Pair(id,DiagonalQuadrature(orth_grid,H_closure_stencils)))
    end
    return Laplace(Δ, H, H⁻¹, Dict(e_pairs), Dict(d_pairs), Dict(Hᵧ_pairs))
end

LazyTensors.range_size(L::Laplace) = LazyTensors.range_size(L.D)
LazyTensors.domain_size(L::Laplace) = LazyTensors.domain_size(L.D)
LazyTensors.apply(L::Laplace, v::AbstractArray, I...) = LazyTensors.apply(L.D,v,I...)

quadrature(L::Laplace) = L.H
export quadrature
inverse_quadrature(L::Laplace) = L.H_inv
export inverse_quadrature
boundary_restriction(L::Laplace,bid::BoundaryIdentifier) = L.e[bid]
export boundary_restriction
normal_derivative(L::Laplace,bid::BoundaryIdentifier) = L.d[bid]
export normal_derivative
boundary_quadrature(L::Laplace,bid::BoundaryIdentifier) = L.H_boundary[bid]
export boundary_quadrature

"""
    laplace(grid::EquidistantGrid{Dim}, inner_stencil, closure_stencils)

Creates the Laplace operator operator `Δ` as a `TensorMapping`

`Δ` approximates the Laplace operator ∑d²/xᵢ² , i = 1,...,`Dim` on `grid`, using
the stencil `inner_stencil` in the interior and a set of stencils `closure_stencils`
for the points in the closure regions.

On a one-dimensional `grid`, `Δ` is a `SecondDerivative`. On a multi-dimensional `grid`, `Δ` is the sum of
multi-dimensional `SecondDerivative`s where the sum is carried out lazily.
"""
function laplace(grid::EquidistantGrid{Dim}, inner_stencil, closure_stencils) where Dim
    Δ = SecondDerivative(grid, inner_stencil, closure_stencils, 1)
    for d = 2:Dim
        Δ += SecondDerivative(grid, inner_stencil, closure_stencils, d)
    end
    return Δ
end
export laplace
