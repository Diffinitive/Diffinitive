# TODO:refactor to take a tuple instead. Convert the tuple to stencil for now. Could probably be refactored using a diagonal operator later.
# How would a block-ip be built? A method on inner_product taking a stencil collection for the closure which then returns a different type of tensormapping

"""
    inner_product(grid::EquidistantGrid, closure_stencils, inner_stencil)

Creates the discrete inner product operator `H` as a `TensorMapping` on an equidistant
grid, defined as `(u,v)  = u'Hv` for grid functions `u,v`.

`inner_product(grid::EquidistantGrid, closure_stencils, inner_stencil)` creates
`H` on `grid` the using a set of stencils `closure_stencils` for the points in
the closure regions and the stencil and `inner_stencil` in the interior.

On a 1-dimensional `grid`, `H` is a `VolumeOperator`. On a N-dimensional
`grid`, `H` is the outer product of the 1-dimensional inner product operators in
each coordinate direction. Also see the documentation of
`SbpOperators.volume_operator(...)` for more details. On a 0-dimensional `grid`,
`H` is a 0-dimensional `IdentityMapping`.
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
