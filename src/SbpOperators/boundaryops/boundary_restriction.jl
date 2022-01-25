"""
    boundary_restriction(grid::EquidistantGrid, closure_stencil::Stencil, boundary::CartesianBoundary)
    boundary_restriction(grid::EquidistantGrid{1}, closure_stencil::Stencil, region::Region)

Creates the boundary restriction operator `e` as a `TensorMapping`

`e` is the restriction of a grid function to the boundary specified by `boundary` or `region` using some `closure_stencil`.
`e'` is the prolongation of a grid function on the boundary to the whole grid using the same `closure_stencil`.
On a one-dimensional `grid`, `e` is a `BoundaryOperator`. On a multi-dimensional `grid`, `e` is the inflation of
a `BoundaryOperator`. Also see the documentation of `SbpOperators.boundary_operator(...)` for more details.
"""
function boundary_restriction(grid::EquidistantGrid, closure_stencil, boundary::CartesianBoundary)
    converted_stencil = convert(Stencil{eltype(grid)}, closure_stencil)
    return SbpOperators.boundary_operator(grid, converted_stencil, boundary)
end
boundary_restriction(grid::EquidistantGrid{1}, closure_stencil, region::Region) = boundary_restriction(grid, closure_stencil, CartesianBoundary{1,typeof(region)}())
