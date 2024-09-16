"""
    inverse_inner_product(grid, ...)

The inverse inner product on a given grid with weights from a stencils set or given
explicitly.
"""
function inverse_inner_product end

"""
    inverse_inner_product(tg::TensorGrid, stencil_set::StencilSet)

The inverse of inner product on `tg`, i.e., the tensor product of the
individual grids' inverse inner products, using weights `H` from `stencil_set`.
"""
function inverse_inner_product(tg::TensorGrid, stencil_set::StencilSet)
    return mapreduce(g->inverse_inner_product(g,stencil_set), ⊗, tg.grids)
end

"""
    inverse_inner_product(g::EquidistantGrid, stencil_set::StencilSet)

The inverse of the inner product on `g` using weights `H` from `stencil_set`.

See also: [`ConstantInteriorScalingOperator`](@ref).
"""
function inverse_inner_product(g::EquidistantGrid, stencil_set::StencilSet)
    interior_weight = parse_scalar(stencil_set["H"]["inner"])
    closure_weights = parse_tuple(stencil_set["H"]["closure"])
    return inverse_inner_product(g, interior_weight, closure_weights)
end

"""
    inverse_inner_product(g::EquidistantGrid, interior_weight, closure_weights)

The inverse inner product on `g` with explicit weights.

See also: [`ConstantInteriorScalingOperator`](@ref).
"""
function inverse_inner_product(g::EquidistantGrid, interior_weight, closure_weights)
    h⁻¹ = inverse_spacing(g)
    return SbpOperators.ConstantInteriorScalingOperator(g, h⁻¹*1/interior_weight, h⁻¹./closure_weights)
end

"""
    inverse_inner_product(g::ZeroDimGrid, stencil_set::StencilSet)

The identity tensor with the correct type parameters.

Implemented to simplify 1D code for SBP operators.
"""
inverse_inner_product(g::ZeroDimGrid, stencil_set::StencilSet) = IdentityTensor{component_type(g)}()

function inverse_inner_product(g::MappedGrid, stencil_set)
    J⁻¹ = map(inv∘sqrt∘det, metric_tensor(g))
    DiagonalTensor(J⁻¹)∘inverse_inner_product(logical_grid(g), stencil_set)
end
