"""
    inner_product(grid::EquidistantGrid, closure_stencils, inner_stencil)

Creates the discrete inner product operator `H` as a `TensorMapping` on an equidistant
grid, defined as `(u,v)  = u'Hv` for grid functions `u,v`.

`inner_product(grid::EquidistantGrid, closure_stencils, inner_stencil)` creates
`H` on `grid` the using a set of stencils `closure_stencils` for the points in
the closure regions and the stencil and `inner_stencil` in the interior. If
`inner_stencil` is omitted a central interior stencil with weight 1 is used.

On a 1-dimensional `grid`, `H` is a `VolumeOperator`. On a N-dimensional
`grid`, `H` is the outer product of the 1-dimensional inner product operators in
each coordinate direction. Also see the documentation of
`SbpOperators.volume_operator(...)` for more details. On a 0-dimensional `grid`,
`H` is a 0-dimensional `IdentityMapping`.
"""
function inner_product(grid::EquidistantGrid, closure_stencils, inner_stencil = CenteredStencil(one(eltype(grid))))
    Hs = ()

    for i ∈ 1:dimension(grid)
        Hs = (Hs..., inner_product(restrict(grid, i), closure_stencils, inner_stencil))
    end

    return foldl(⊗, Hs)
end
export inner_product

function inner_product(grid::EquidistantGrid{1}, closure_stencils, inner_stencil = CenteredStencil(one(eltype(grid))))
    h = spacing(grid)
    H = SbpOperators.volume_operator(grid, scale(inner_stencil,h[1]), scale.(closure_stencils,h[1]), even, 1)
    return H
end

inner_product(grid::EquidistantGrid{0}, closure_stencils, inner_stencil) = IdentityMapping{eltype(grid)}()
