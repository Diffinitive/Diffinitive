export first_derivative

"""
    first_derivative(grid::EquidistantGrid, inner_stencil, closure_stencils, direction)

Creates the first-derivative operator `D1` as a `TensorMapping`

`D1` approximates the first-derivative d/dξ on `grid` along the coordinate dimension specified by
`direction`, using the stencil `inner_stencil` in the interior and a set of stencils `closure_stencils`
for the points in the closure regions.

On a one-dimensional `grid`, `D1` is a `VolumeOperator`. On a multi-dimensional `grid`, `D1` is the outer product of the
one-dimensional operator with the `IdentityMapping`s in orthogonal coordinate dirrections.

See also: [`SbpOperators.volume_operator`](@ref).
"""
function first_derivative(grid::EquidistantGrid, inner_stencil, closure_stencils, direction)
    h_inv = inverse_spacing(grid)[direction]
    return SbpOperators.volume_operator(grid, scale(inner_stencil,h_inv), scale.(closure_stencils,h_inv), odd, direction)
end
first_derivative(grid::EquidistantGrid{1}, inner_stencil, closure_stencils) = first_derivative(grid,inner_stencil,closure_stencils,1)

