"""
    quadrature(grid::EquidistantGrid, inner_stencil, closure_stencils)
    quadrature(grid::EquidistantGrid, closure_stencils)

Creates the quadrature operator `H` as a `TensorMapping`

`H` approximiates the integral operator on `grid` the using the stencil
`inner_stencil` in the interior and a set of stencils `closure_stencils`
for the points in the closure regions. If `inner_stencil` is omitted a central
interior stencil with weight 1 is used.

On a one-dimensional `grid`, `H` is a `VolumeOperator`. On a multi-dimensional
`grid`, `H` is the outer product of the 1-dimensional quadrature operators in
each coordinate direction. Also see the documentation of
`SbpOperators.volume_operator(...)` for more details. On 0-dimensional `grid`,
`H` is a 0-dimensional `IdentityMapping`.
"""
function quadrature(grid::EquidistantGrid, inner_stencil, closure_stencils)
    h = spacing(grid)
    H = SbpOperators.volume_operator(grid, scale(inner_stencil,h[1]), scale.(closure_stencils,h[1]), even, 1)
    for i ∈ 2:dimension(grid)
        Hᵢ = SbpOperators.volume_operator(grid, scale(inner_stencil,h[i]), scale.(closure_stencils,h[i]), even, i)
        H = H∘Hᵢ
    end
    return H
end
export quadrature

quadrature(grid::EquidistantGrid{0,T}, inner_stencil, closure_stencils) where T = IdentityMapping{T}()
#TODO:  Consider changing the interface of volume_operator to volume_operator(grid,closure_stencils,inner_stencil)
#       in order to allow for having quadrature(grid, closure_stencils, inner_stencil = CenteredStencil(one(T)))
#       Then the below function can be removed.
function quadrature(grid::EquidistantGrid{Dim,T}, closure_stencils) where {Dim,T}
    inner_stencil = CenteredStencil(one(T))
    return quadrature(grid, inner_stencil, closure_stencils)
end
