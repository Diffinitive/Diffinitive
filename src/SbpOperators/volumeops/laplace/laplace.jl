"""
    Laplace{T, Dim, ...} <: TensorMapping{T,Dim,Dim}
    Laplace(grid::EquidistantGrid, fn; order)

Implements the Laplace operator, approximating ∑d²/xᵢ² , i = 1,...,`Dim` as a
`TensorMapping`. Additionally, `Laplace` stores the inner product and boundary
operators relevant for constructing a SBP finite difference scheme as `TensorMapping`s.

Laplace(grid::EquidistantGrid, fn; order) creates the Laplace operator on an
equidistant grid, where the operators are read from TOML. The differential operator
is created using `laplace(grid,...)`.
"""
struct Laplace{T, Dim, Rb, TMdiffop<:TensorMapping{T,Dim,Dim}, # Differential operator
                           TMipop<:TensorMapping{T,Dim,Dim}, # Inner product operator
                           TMbop<:TensorMapping{T,Rb,Dim}, # Boundary operator
                           TMbqop<:TensorMapping{T,Rb,Rb}, # Boundary quadrature
                           BID<:BoundaryIdentifier} <: TensorMapping{T,Dim,Dim}
    D::TMdiffop # Difference operator
    H::TMipop # Inner product operator
    H_inv::TMipop # Inverse inner product operator
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
    H =  inner_product(grid, H_closure_stencils)
    H⁻¹ =  inverse_inner_product(grid, H_closure_stencils)

    # Boundary operator - id pairs
    ids = boundary_identifiers(grid)
    n_ids = length(ids)
    e_pairs = ntuple(i -> Pair(ids[i],boundary_restriction(grid,e_closure_stencil,ids[i])),n_ids)
    d_pairs = ntuple(i -> Pair(ids[i],normal_derivative(grid,d_closure_stencil,ids[i])),n_ids)
    Hᵧ_pairs = ntuple(i -> Pair(ids[i],inner_product(boundary_grid(grid,ids[i]),H_closure_stencils)),n_ids)

    return Laplace(Δ, H, H⁻¹, Dict(e_pairs), Dict(d_pairs), Dict(Hᵧ_pairs))
end

LazyTensors.range_size(L::Laplace) = LazyTensors.range_size(L.D)
LazyTensors.domain_size(L::Laplace) = LazyTensors.domain_size(L.D)
LazyTensors.apply(L::Laplace, v::AbstractArray, I...) = LazyTensors.apply(L.D,v,I...)

inner_product(L::Laplace) = L.H
export inner_product
inverse_inner_product(L::Laplace) = L.H_inv
export inverse_inner_product
boundary_restriction(L::Laplace,bid::BoundaryIdentifier) = L.e[bid]
export boundary_restriction
normal_derivative(L::Laplace,bid::BoundaryIdentifier) = L.d[bid]
export normal_derivative
# TODO: boundary_inner_product?
boundary_quadrature(L::Laplace,bid::BoundaryIdentifier) = L.H_boundary[bid]
export boundary_quadrature

"""
    laplace(grid::EquidistantGrid, inner_stencil, closure_stencils)

Creates the Laplace operator operator `Δ` as a `TensorMapping`

`Δ` approximates the Laplace operator ∑d²/xᵢ² , i = 1,...,N on the N-dimensional
`grid`, using the stencil `inner_stencil` in the interior and a set of stencils
`closure_stencils` for the points in the closure regions.

On a one-dimensional `grid`, `Δ` is equivalent to `second_derivative`. On a
multi-dimensional `grid`, `Δ` is the sum of multi-dimensional `second_derivative`s
where the sum is carried out lazily.
"""
function laplace(grid::EquidistantGrid, inner_stencil, closure_stencils)
    Δ = second_derivative(grid, inner_stencil, closure_stencils, 1)
    for d = 2:dimension(grid)
        Δ += second_derivative(grid, inner_stencil, closure_stencils, d)
    end
    return Δ
end
export laplace
