"""
    boundary_restriction(g, closure_stencil::Stencil, boundary)

Creates boundary restriction operators `e` as `LazyTensor`s on `boundary`

`e` is the restriction of a grid function to `boundary` using a `Stencil` `closure_stencil`.
`e'` is the prolongation of a grid function on `boundary` to the whole grid using the same `closure_stencil`.
On a one-dimensional grid, `e` is a `BoundaryOperator`. On a multi-dimensional grid, `e` is the inflation of
a `BoundaryOperator`.

See also: [`BoundaryOperator`](@ref), [`LazyTensors.inflate`](@ref).
"""
#TODO: Check docstring
function boundary_restriction(g::TensorGrid, stencil_set::StencilSet, boundary::TensorGridBoundary)
    op = boundary_restriction(g.grids[grid_id(boundary)], stencil_set, boundary_id(boundary))
    return LazyTensors.inflate(op, size(g), grid_id(boundary))
end

function boundary_restriction(g::EquidistantGrid, stencil_set::StencilSet, boundary)
    closure_stencil = parse_stencil(stencil_set["e"]["closure"])
    converted_stencil = convert(Stencil{eltype(g)}, closure_stencil)
    return BoundaryOperator(g, converted_stencil, boundary)
end
