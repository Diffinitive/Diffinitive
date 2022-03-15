"""
    inverse_inner_product(grid::EquidistantGrid, interior_weight, closure_weights)

Constructs the inverse inner product operator `H⁻¹` as a `TensorMapping` using
the weights of `H`, `interior_weight`, `closure_weights`. `H⁻¹` is inverse of
the inner product operator `H`.

On a 1-dimensional grid, `H⁻¹` is a `ConstantInteriorScalingOperator`. On an
N-dimensional grid, `H⁻¹` is the outer product of the 1-dimensional inverse
inner product operators for each coordinate direction. On a 0-dimensional
`grid`, `H⁻¹` is a 0-dimensional `IdentityMapping`. 

See also: [`ConstantInteriorScalingOperator`](@ref).
"""
function inverse_inner_product(grid::EquidistantGrid, interior_weight, closure_weights)
    H⁻¹s = ()

    for i ∈ 1:dimension(grid)
        H⁻¹s = (H⁻¹s..., inverse_inner_product(restrict(grid, i), interior_weight, closure_weights))
    end

    return foldl(⊗, H⁻¹s)
end

function inverse_inner_product(grid::EquidistantGrid{1}, interior_weight, closure_weights)
    h⁻¹ = inverse_spacing(grid)[1]
    H⁻¹ = SbpOperators.ConstantInteriorScalingOperator(grid, h⁻¹*1/interior_weight, h⁻¹./closure_weights)
    return H⁻¹
end

inverse_inner_product(grid::EquidistantGrid{0}, interior_weight, closure_weights) = IdentityMapping{eltype(grid)}()

"""
    inverse_inner_product(grid, stencil_set)

Creates a `inverse_inner_product` operator on `grid` given a parsed TOML
`stencil_set`.
"""
function inverse_inner_product(grid, stencil_set)
    inner_stencil = parse_scalar(stencil_set["H"]["inner"])
    closure_stencils = parse_tuple(stencil_set["H"]["closure"])
    return inverse_inner_product(grid, inner_stencil, closure_stencils)
end
