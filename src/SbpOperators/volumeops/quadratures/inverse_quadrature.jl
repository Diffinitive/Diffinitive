
"""
    InverseQuadrature(grid::EquidistantGrid, inv_inner_stencil, inv_closure_stencils)

Creates the inverse `H⁻¹` of the quadrature operator as a `TensorMapping`

The inverse quadrature approximates the integral operator on the grid using
`inv_inner_stencil` in the interior and a set of stencils `inv_closure_stencils`
for the points in the closure regions.

On a one-dimensional `grid`, `H⁻¹` is a `VolumeOperator`. On a multi-dimensional
`grid`, `H` is the outer product of the 1-dimensional inverse quadrature operators in
each coordinate direction. Also see the documentation of
`SbpOperators.volume_operator(...)` for more details.
"""
function InverseQuadrature(grid::EquidistantGrid{Dim}, inv_inner_stencil, inv_closure_stencils) where Dim
    h⁻¹ = inverse_spacing(grid)
    H⁻¹ = SbpOperators.volume_operator(grid,scale(inv_inner_stencil,h⁻¹[1]),scale.(inv_closure_stencils,h⁻¹[1]),even,1)
    for i ∈ 2:Dim
        Hᵢ⁻¹ = SbpOperators.volume_operator(grid,scale(inv_inner_stencil,h⁻¹[i]),scale.(inv_closure_stencils,h⁻¹[i]),even,i)
        H⁻¹ = H⁻¹∘Hᵢ⁻¹
    end
    return H⁻¹
end
export InverseQuadrature

"""
    InverseDiagonalQuadrature(grid::EquidistantGrid, closure_stencils)

Creates the inverse of the diagonal quadrature operator defined by the inner stencil
1/h and a set of 1-element closure stencils in `closure_stencils`. Note that
the closure stencils are those of the quadrature operator (and not the inverse).
"""
function InverseDiagonalQuadrature(grid::EquidistantGrid, closure_stencils::NTuple{M,Stencil{T,1}}) where {T,M}
    inv_inner_stencil = Stencil(one(T), center=1)
    inv_closure_stencils = reciprocal_stencil.(closure_stencils)
    return InverseQuadrature(grid, inv_inner_stencil, inv_closure_stencils)
end
export InverseDiagonalQuadrature

reciprocal_stencil(s::Stencil{T}) where T = Stencil(s.range,one(T)./s.weights)
