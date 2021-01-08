"""
    Laplace(grid::EquidistantGrid{Dim}, inner_stencil, closure_stencils)

Creates the Laplace operator operator `Δ` as a `TensorMapping`

`Δ` approximates the Laplace operator ∑d²/xᵢ² , i = 1,...,`Dim` on `grid`, using
the stencil `inner_stencil` in the interior and a set of stencils `closure_stencils`
for the points in the closure regions.

On a one-dimensional `grid`, `Δ` is a `SecondDerivative`. On a multi-dimensional `grid`, `Δ` is the sum of
multi-dimensional `SecondDerivative`s where the sum is carried out lazily.
"""
function Laplace(grid::EquidistantGrid{Dim}, inner_stencil, closure_stencils) where Dim
    Δ = SecondDerivative(grid, inner_stencil, closure_stencils, 1)
    for d = 2:Dim
        Δ += SecondDerivative(grid, inner_stencil, closure_stencils, d)
    end
    return Δ
end
export Laplace
