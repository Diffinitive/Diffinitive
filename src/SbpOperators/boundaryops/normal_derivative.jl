"""
    normal_derivative(grid, closure_stencil::Stencil, boundary)

Creates the normal derivative boundary operator `d` as a `LazyTensor`

`d` computes the normal derivative of a grid function  on `boundary` a `Stencil` `closure_stencil`.
`d'` is the prolongation of the normal derivative of a grid function to the whole grid using the same `closure_stencil`.
On a one-dimensional `grid`, `d` is a `BoundaryOperator`. On a multi-dimensional `grid`, `d` is the inflation of
a `BoundaryOperator`.

See also: [`BoundaryOperator`](@ref), [`LazyTensors.inflate`](@ref).
"""
function normal_derivative(grid, closure_stencil, boundary)
    direction = dim(boundary)
    h_inv = inverse_spacing(grid)[direction]

    op = BoundaryOperator(restrict(grid, dim(boundary)), scale(closure_stencil,h_inv), region(boundary))
    return LazyTensors.inflate(op, size(grid), dim(boundary))
end

"""
    normal_derivative(grid, stencil_set, boundary)

Creates a `normal_derivative` operator on `grid` given a `stencil_set`.
"""
normal_derivative(grid, stencil_set::StencilSet, boundary) = normal_derivative(grid, parse_stencil(stencil_set["d1"]["closure"]), boundary)
