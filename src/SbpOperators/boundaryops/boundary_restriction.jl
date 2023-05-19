"""
    boundary_restriction(g, stencil_set::StencilSet, boundary)
    boundary_restriction(g::TensorGrid, stencil_set::StencilSet, boundary::TensorGridBoundary)
    boundary_restriction(g::EquidistantGrid, stencil_set::StencilSet, boundary)

Creates boundary restriction operators `e` as `LazyTensor`s on `boundary`

`e` restricts a grid function on `g` to `boundary` using the 'e' stencil
in `stencil_set`. `e'` prolongates a grid function on
`boundary` to the whole grid using the same stencil. On a one-dimensional
grid, `e` is a `BoundaryOperator`. On a multi-dimensional grid, `e` is the
inflation of a `BoundaryOperator`.

See also: [`BoundaryOperator`](@ref), [`LazyTensors.inflate`](@ref).
"""
function boundary_restriction end

function boundary_restriction(g::TensorGrid, stencil_set::StencilSet, boundary::TensorGridBoundary)
    op = boundary_restriction(g.grids[grid_id(boundary)], stencil_set, boundary_id(boundary))
    return LazyTensors.inflate(op, size(g), grid_id(boundary))
end

function boundary_restriction(g::EquidistantGrid, stencil_set::StencilSet, boundary)
    closure_stencil = parse_stencil(stencil_set["e"]["closure"])
    converted_stencil = convert(Stencil{eltype(g)}, closure_stencil)
    return BoundaryOperator(g, converted_stencil, boundary)
end
