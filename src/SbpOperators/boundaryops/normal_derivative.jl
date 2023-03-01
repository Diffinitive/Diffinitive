"""
    normal_derivative(g, closure_stencil::Stencil, boundary)

Creates the normal derivative boundary operator `d` as a `LazyTensor`

`d` computes the normal derivative of a grid function  on `boundary` a `Stencil` `closure_stencil`.
`d'` is the prolongation of the normal derivative of a grid function to the whole grid using the same `closure_stencil`.
On a one-dimensional `grid`, `d` is a `BoundaryOperator`. On a multi-dimensional `grid`, `d` is the inflation of
a `BoundaryOperator`.

See also: [`BoundaryOperator`](@ref), [`LazyTensors.inflate`](@ref).
"""
#TODO: Check docstring
function normal_derivative(g::TensorGrid, stencil_set::StencilSet, boundary::TensorGridBoundary)
    op = normal_derivative(g.grids[grid_id(boundary)], stencil_set, boundary_id(boundary))
    return LazyTensors.inflate(op, size(g), grid_id(boundary))
end

function normal_derivative(g::EquidistantGrid, stencil_set::StencilSet, boundary)
    closure_stencil = parse_stencil(stencil_set["d1"]["closure"])
    h_inv = inverse_spacing(g)

    scaled_stencil = scale(closure_stencil,h_inv)
    return BoundaryOperator(g, scaled_stencil, boundary)
end
