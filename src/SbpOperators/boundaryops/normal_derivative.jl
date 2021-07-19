"""
    normal_derivative(grid::EquidistantGrid, closure_stencil::Stencil, boundary::CartesianBoundary)
    normal_derivative(grid::EquidistantGrid{1}, closure_stencil::Stencil, region::Region)

Creates the normal derivative boundary operator `d` as a `TensorMapping`

`d` is the normal derivative of a grid function at the boundary specified by `boundary` or `region` using some `closure_stencil`.
`d'` is the prolongation of the normal derivative of a grid function to the whole grid using the same `closure_stencil`.
On a one-dimensional `grid`, `d` is a `BoundaryOperator`. On a multi-dimensional `grid`, `d` is the inflation of
a `BoundaryOperator`. Also see the documentation of `SbpOperators.boundary_operator(...)` for more details.
"""
function normal_derivative(grid::EquidistantGrid, closure_stencil, boundary::CartesianBoundary)
    direction = dim(boundary)
    h_inv = inverse_spacing(grid)[direction]
    return SbpOperators.boundary_operator(grid, scale(closure_stencil,h_inv), boundary)
end
normal_derivative(grid::EquidistantGrid{1}, closure_stencil, region::Region) = normal_derivative(grid, closure_stencil, CartesianBoundary{1,typeof(region)}())
export normal_derivative
