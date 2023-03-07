"""
    inverse_inner_product(grid::EquidistantGrid, interior_weight, closure_weights)

Constructs the inverse inner product operator `H⁻¹` as a `LazyTensor` using
the weights of `H`, `interior_weight`, `closure_weights`. `H⁻¹` is inverse of
the inner product operator `H`.

On a 1-dimensional grid, `H⁻¹` is a `ConstantInteriorScalingOperator`. On an
N-dimensional grid, `H⁻¹` is the outer product of the 1-dimensional inverse
inner product operators for each coordinate direction. On a 0-dimensional
`grid`, `H⁻¹` is a 0-dimensional `IdentityTensor`.

See also: [`ConstantInteriorScalingOperator`](@ref).
"""
function inverse_inner_product(tg::TensorGrid, stencil_set::StencilSet)
    return mapreduce(g->inverse_inner_product(g,stencil_set), ⊗, tg.grids)
end

function inverse_inner_product(g::EquidistantGrid, stencil_set::StencilSet)
    interior_weight = parse_scalar(stencil_set["H"]["inner"])
    closure_weights = parse_tuple(stencil_set["H"]["closure"])

    h⁻¹ = inverse_spacing(g)
    return SbpOperators.ConstantInteriorScalingOperator(g, h⁻¹*1/interior_weight, h⁻¹./closure_weights)
end

inverse_inner_product(g::ZeroDimGrid, stencil_set::StencilSet) = IdentityTensor{component_type(g)}()
