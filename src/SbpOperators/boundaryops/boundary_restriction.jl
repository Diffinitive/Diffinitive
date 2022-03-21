# TODO: The type parameter closure_stencil::Stencil is required since there isnt any suitable type
# for stencil_set. We should consider adding type ::StencilSet and dispatch on that instead.
# The same goes for other operators
"""
    boundary_restriction(grid, closure_stencil::Stencil, boundary)

Creates boundary restriction operators `e` as `LazyTensor`s on `boundary`

`e` is the restriction of a grid function to `boundary` using a `Stencil` `closure_stencil`.
`e'` is the prolongation of a grid function on `boundary` to the whole grid using the same `closure_stencil`.
On a one-dimensional `grid`, `e` is a `BoundaryOperator`. On a multi-dimensional `grid`, `e` is the inflation of
a `BoundaryOperator`.

See also: [`boundary_operator`](@ref).
"""
function boundary_restriction(grid, closure_stencil::Stencil, boundary)
    converted_stencil = convert(Stencil{eltype(grid)}, closure_stencil)

    op = BoundaryOperator(restrict(grid, dim(boundary)), converted_stencil, region(boundary))
    return LazyTensors.inflate(op, size(grid), dim(boundary))
end

"""
    boundary_restriction(grid, stencil_set, boundary)

Creates a `boundary_restriction` operator on `grid` given a parsed TOML
`stencil_set`.
"""
boundary_restriction(grid, stencil_set, boundary) = boundary_restriction(grid, parse_stencil(stencil_set["e"]["closure"]), boundary)
