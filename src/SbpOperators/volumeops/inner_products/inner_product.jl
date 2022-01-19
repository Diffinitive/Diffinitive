"""
    inner_product(grid::EquidistantGrid, interior_weight, closure_weights)

Creates the discrete inner product operator `H` as a `TensorMapping` on an
equidistant grid, defined as `(u,v)  = u'Hv` for grid functions `u,v`.

`inner_product` creates `H` on `grid` using the `interior_weight` for the
interior points and the `closure_weights` for the points close to the
boundary.

On a 1-dimensional grid, `H` is a `ConstantInteriorScalingOperator`. On a
N-dimensional grid, `H` is the outer product of the 1-dimensional inner
product operators for each coordinate direction. Also see the documentation of
On a 0-dimensional grid, `H` is a 0-dimensional `IdentityMapping`.
"""
function inner_product(grid::EquidistantGrid, interior_weight, closure_weights)
    Hs = ()

    for i ∈ 1:dimension(grid)
        Hs = (Hs..., inner_product(restrict(grid, i), interior_weight, closure_weights))
    end

    return foldl(⊗, Hs)
end
export inner_product

function inner_product(grid::EquidistantGrid{1}, interior_weight, closure_weights)
    h = spacing(grid)[1]

    H = SbpOperators.ConstantInteriorScalingOperator(grid, h*interior_weight, h.*closure_weights)
    return H
end

inner_product(grid::EquidistantGrid{0}, interior_weight, closure_weights) = IdentityMapping{eltype(grid)}()
