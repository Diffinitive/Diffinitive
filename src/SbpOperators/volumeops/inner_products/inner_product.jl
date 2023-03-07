"""
    inner_product(grid::EquidistantGrid, interior_weight, closure_weights)

Creates the discrete inner product operator `H` as a `LazyTensor` on an
equidistant grid, defined as `(u,v)  = u'Hv` for grid functions `u,v`.

`inner_product` creates `H` on `grid` using the `interior_weight` for the
interior points and the `closure_weights` for the points close to the
boundary.

On a 1-dimensional grid, `H` is a `ConstantInteriorScalingOperator`. On a
N-dimensional grid, `H` is the outer product of the 1-dimensional inner
product operators for each coordinate direction. On a 0-dimensional grid,
`H` is a 0-dimensional `IdentityTensor`.

See also: [`ConstantInteriorScalingOperator`](@ref).
"""
function inner_product(tg::TensorGrid, stencil_set::StencilSet)
    return mapreduce(g->inner_product(g,stencil_set), ⊗, tg.grids)
end

function inner_product(g::EquidistantGrid, stencil_set::StencilSet)
    interior_weight = parse_scalar(stencil_set["H"]["inner"])
    closure_weights = parse_tuple(stencil_set["H"]["closure"])

    h = spacing(g)
    return SbpOperators.ConstantInteriorScalingOperator(g, h*interior_weight, h.*closure_weights)
end

inner_product(g::ZeroDimGrid, stencil_set::StencilSet) = IdentityTensor{component_type(g)}()

