"""
    second_derivative(grid::EquidistantGrid, inner_stencil, closure_stencils, direction)

Creates the second-derivative operator `D2` as a `TensorMapping`

`D2` approximates the second-derivative d²/dξ² on `grid` along the coordinate dimension specified by
`direction`, using the stencil `inner_stencil` in the interior and a set of stencils `closure_stencils`
for the points in the closure regions.

On a one-dimensional `grid`, `D2` is a `VolumeOperator`. On a multi-dimensional `grid`, `D2` is the outer product of the
one-dimensional operator with the `IdentityMapping`s in orthogonal coordinate dirrections.

See also: [`volume_operator`](@ref).
"""
function second_derivative(grid::EquidistantGrid, inner_stencil, closure_stencils, direction)
    h_inv = inverse_spacing(grid)[direction]
    return SbpOperators.volume_operator(grid, scale(inner_stencil,h_inv^2), scale.(closure_stencils,h_inv^2), even, direction)
end
second_derivative(grid::EquidistantGrid{1}, inner_stencil, closure_stencils) = second_derivative(grid,inner_stencil,closure_stencils,1)

# REVIEW: Stencil set method?
