"""
    inner_product(grid, ...)

The inner product on a given grid with weights from a stencils set or given
explicitly.
"""
function inner_product end

"""
    inner_product(tg::TensorGrid, stencil_set::StencilSet)

The inner product on `tg`, i.e., the tensor product of the
individual grids' inner products, using weights `H` from `stencil_set`.
"""
function inner_product(tg::TensorGrid, stencil_set::StencilSet)
    return mapreduce(g->inner_product(g,stencil_set), ⊗, tg.grids)
end

"""
    inner_product(g::EquidistantGrid, stencil_set::StencilSet)

The inner product on `g` using weights `H` from `stencil_set`.

See also: [`ConstantInteriorScalingOperator`](@ref).
"""
function inner_product(g::EquidistantGrid, stencil_set::StencilSet)
    interior_weight = parse_scalar(stencil_set["H"]["inner"])
    closure_weights = parse_tuple(stencil_set["H"]["closure"])
    return inner_product(g, interior_weight, closure_weights)
end

"""
    inner_product(g::EquidistantGrid, interior_weight, closure_weights)

The inner product on `g` with explicit weights.

See also: [`ConstantInteriorScalingOperator`](@ref).
"""
function inner_product(g::EquidistantGrid, interior_weight, closure_weights)
    h = spacing(g)
    return SbpOperators.ConstantInteriorScalingOperator(g, h*interior_weight, h.*closure_weights)
end

"""
    inner_product(g::ZeroDimGrid, stencil_set::StencilSet)

The identity tensor with the correct type parameters.

Implemented to simplify 1D code for SBP operators.
"""
inner_product(g::ZeroDimGrid, stencil_set::StencilSet) = IdentityTensor{component_type(g)}()


function inner_product(g::MappedGrid, stencil_set)
    J = map(sqrt∘det, metric_tensor(g))
    DiagonalTensor(J)∘inner_product(logicalgrid(g), stencil_set)
end
