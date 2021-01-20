"""
    SecondDerivative(grid::EquidistantGrid{Dim}, inner_stencil, closure_stencils, direction)
    SecondDerivative(grid::EquidistantGrid{1}, inner_stencil, closure_stencils)

Creates the second-derivative operator `D2` as a `TensorMapping`

`D2` approximates the second-derivative d²/dξ² on `grid` along the coordinate dimension specified by
`direction`, using the stencil `inner_stencil` in the interior and a set of stencils `closure_stencils`
for the points in the closure regions.

On a one-dimensional `grid`, `D2` is a `VolumeOperator`. On a multi-dimensional `grid`, `D2` is the outer product of the
one-dimensional operator with the `IdentityMapping`s in orthogonal coordinate dirrections.
Also see the documentation of `SbpOperators.volume_operator(...)` for more details.
"""
function SecondDerivative(grid::EquidistantGrid{Dim}, inner_stencil, closure_stencils, direction) where Dim
    h_inv = inverse_spacing(grid)[direction]
    return SbpOperators.volume_operator(grid, scale(inner_stencil,h_inv^2), scale.(closure_stencils,h_inv^2), even, direction)
end
SecondDerivative(grid::EquidistantGrid{1}, inner_stencil, closure_stencils) = SecondDerivative(grid,inner_stencil,closure_stencils,1)
export SecondDerivative
