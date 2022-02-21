"""
    normal_derivative(grid, closure_stencil::Stencil, boundary)

Creates the normal derivative boundary operator `d` as a `TensorMapping`

`d` computes the normal derivative of a grid function  on `boundary` a `Stencil` `closure_stencil`.
`d'` is the prolongation of the normal derivative of a grid function to the whole grid using the same `closure_stencil`.
On a one-dimensional `grid`, `d` is a `BoundaryOperator`. On a multi-dimensional `grid`, `d` is the inflation of
a `BoundaryOperator`. See also [`SbpOperators.boundary_operator`](@ref).
"""
function normal_derivative(grid, closure_stencil::Stencil, boundary)
    direction = dim(boundary)
    h_inv = inverse_spacing(grid)[direction]
    return SbpOperators.boundary_operator(grid, scale(closure_stencil,h_inv), boundary)
end

"""
    normal_derivative(grid, stencil_set, boundary)

Creates a `normal_derivative` operator on `grid` given a parsed TOML
`stencil_set`.
"""
normal_derivative(grid, stencil_set, boundary) = normal_derivative(grid, parse_stencil(stencil_set["e"]["closure"]), boundary)
