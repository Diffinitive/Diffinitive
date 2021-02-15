"""
    boundary_restriction(grid::EquidistantGrid, closure_stencil::Stencil, boundary::CartesianBoundary)
    boundary_restriction(grid::EquidistantGrid{1}, closure_stencil::Stencil, region::Region)

Creates the boundary restriction operator `e` as a `TensorMapping`

`e` is the restriction of a grid function to the boundary specified by `boundary` or `region` using some `closure_stencil`.
`e'` is the prolongation of a grid function on the boundary to the whole grid using the same `closure_stencil`.
On a one-dimensional `grid`, `e` is a `BoundaryOperator`. On a multi-dimensional `grid`, `e` is the inflation of
a `BoundaryOperator`. Also see the documentation of `SbpOperators.boundary_operator(...)` for more details.
"""
boundary_restriction(grid::EquidistantGrid, closure_stencil::Stencil, boundary::CartesianBoundary) = SbpOperators.boundary_operator(grid, closure_stencil, boundary)
boundary_restriction(grid::EquidistantGrid{1}, closure_stencil::Stencil, region::Region) = boundary_restriction(grid, closure_stencil, CartesianBoundary{1,typeof(region)}())

export boundary_restriction
