"""
    first_derivative(grid::EquidistantGrid, inner_stencil, closure_stencils, direction)

Creates the first-derivative operator `D1` as a `LazyTensor`

`D1` approximates the first-derivative d/dξ on `grid` along the coordinate dimension specified by
`direction`, using the stencil `inner_stencil` in the interior and a set of stencils `closure_stencils`
for the points in the closure regions.

On a one-dimensional `grid`, `D1` is a `VolumeOperator`. On a multi-dimensional `grid`, `D1` is the inflation of
a `VolumeOperator`.

See also: [`VolumeOperator`](@ref), [`LazyTensors.inflate`](@ref).
"""
function first_derivative(grid::EquidistantGrid, inner_stencil, closure_stencils, direction)
    h_inv = inverse_spacing(grid)[direction]

    D₁ = VolumeOperator(restrict(grid, direction), scale(inner_stencil,h_inv), scale.(closure_stencils,h_inv), odd)
    return LazyTensors.inflate(D₁, size(grid), direction)
end


"""
    first_derivative(grid, inner_stencil, closure_stencils)

Creates a `first_derivative` operator on a 1D `grid` given `inner_stencil` and `closure_stencils`.
"""
first_derivative(grid::EquidistantGrid{1}, inner_stencil::Stencil, closure_stencils) = first_derivative(grid, inner_stencil, closure_stencils, 1)


"""
    first_derivative(grid, stencil_set::StencilSet, direction)

Creates a `first_derivative` operator on `grid` along coordinate dimension `direction` given a `stencil_set`.
"""
function first_derivative(grid::EquidistantGrid, stencil_set::StencilSet, direction)
    inner_stencil = parse_stencil(stencil_set["D1"]["inner_stencil"])
    closure_stencils = parse_stencil.(stencil_set["D1"]["closure_stencils"])
    first_derivative(grid,inner_stencil,closure_stencils,direction);
end


"""
    first_derivative(grid, stencil_set)

Creates a `first_derivative` operator on a 1D `grid` given a `stencil_set`.
"""
first_derivative(grid::EquidistantGrid{1}, stencil_set::StencilSet) = first_derivative(grid, stencil_set, 1)
