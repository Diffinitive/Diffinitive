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


"""
    second_derivative(grid, inner_stencil, closure_stencils)

Creates a `second_derivative` operator on a 1D `grid` given `inner_stencil` and `closure_stencils`.
"""
second_derivative(grid::EquidistantGrid{1}, inner_stencil::Stencil, closure_stencils) = second_derivative(grid, inner_stencil, closure_stencils,1)


"""
    second_derivative(grid, stencil_set, direction)

Creates a `second_derivative` operator on `grid` along coordinate dimension `direction` given a `stencil_set`.
"""
function second_derivative(grid::EquidistantGrid, stencil_set::StencilSet, direction)
    inner_stencil = parse_stencil(stencil_set["D2"]["inner_stencil"])
    closure_stencils = parse_stencil.(stencil_set["D2"]["closure_stencils"])
    second_derivative(grid,inner_stencil,closure_stencils,direction);
end


"""
    second_derivative(grid, stencil_set)

Creates a `second_derivative` operator on a 1D `grid` given a `stencil_set`.
"""
second_derivative(grid::EquidistantGrid{1}, stencil_set::StencilSet) = second_derivative(grid, stencil_set, 1)