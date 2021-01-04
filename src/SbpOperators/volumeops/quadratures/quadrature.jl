"""
    Quadrature(grid::EquidistantGrid, inner_stencil, closure_stencils)

Creates the quadrature operator `H` as a `TensorMapping`

The quadrature approximates the integral operator on the grid using
`inner_stencil` in the interior and a set of stencils `closure_stencils`
for the points in the closure regions.

On a one-dimensional `grid`, `H` is a `VolumeOperator`. On a multi-dimensional
`grid`, `H` is the outer product of the 1-dimensional quadrature operators in
each  coordinate direction. Also see the documentation of
`SbpOperators.volume_operator(...)` for more details.
"""
function Quadrature(grid::EquidistantGrid{Dim}, inner_stencil, closure_stencils) where Dim
    h = spacing(grid)
    H = SbpOperators.volume_operator(grid, scale(inner_stencil,h[1]), scale.(closure_stencils,h[1]), even, 1)
    for i ∈ 2:Dim
        Hᵢ = SbpOperators.volume_operator(grid, scale(inner_stencil,h[i]), scale.(closure_stencils,h[i]), even, i)
        H = H∘Hᵢ
    end
    return H
end
export Quadrature

"""
    DiagonalQuadrature(grid::EquidistantGrid, closure_stencils)

Creates the quadrature operator with the inner stencil 1/h and 1-element sized
closure stencils (i.e the operator is diagonal)
"""
function DiagonalQuadrature(grid::EquidistantGrid, closure_stencils::NTuple{M,Stencil{T,1}}) where {M,T}
    inner_stencil = Stencil(Tuple{T}(1),center=1)
    return Quadrature(grid, inner_stencil, closure_stencils)
end
export DiagonalQuadrature
