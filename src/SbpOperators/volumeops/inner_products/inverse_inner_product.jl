"""
    inverse_inner_product(grid::EquidistantGrid, inv_inner_stencil, inv_closure_stencils)
    inverse_inner_product(grid::EquidistantGrid, closure_stencils::NTuple{M,Stencil{T,1}})

Creates the inverse inner product operator `H⁻¹` as a `TensorMapping` on an
equidistant grid. `H⁻¹` is defined implicitly by `H⁻¹∘H = I`, where
`H` is the corresponding inner product operator and `I` is the `IdentityMapping`.

`inverse_inner_product(grid::EquidistantGrid, inv_inner_stencil, inv_closure_stencils)`
constructs `H⁻¹` using a set of stencils `inv_closure_stencils` for the points
in the closure regions and the stencil `inv_inner_stencil` in the interior. If
`inv_closure_stencils` is omitted, a central interior stencil with weight 1 is used.

`inverse_inner_product(grid::EquidistantGrid, closure_stencils::NTuple{M,Stencil{T,1}})`
constructs a diagonal inverse inner product operator where `closure_stencils` are the
closure stencils of `H` (not `H⁻¹`!).

On a 1-dimensional `grid`, `H⁻¹` is a `VolumeOperator`. On a N-dimensional
`grid`, `H⁻¹` is the outer product of the 1-dimensional inverse inner product
operators in each coordinate direction. Also see the documentation of
`SbpOperators.volume_operator(...)` for more details. On a 0-dimensional `grid`,
`H⁻¹` is a 0-dimensional `IdentityMapping`.
"""
function inverse_inner_product(grid::EquidistantGrid, inv_closure_stencils, inv_inner_stencil = CenteredStencil(one(eltype(grid))))
    H⁻¹s = ()

    for i ∈ 1:dimension(grid)
        H⁻¹s = (H⁻¹s..., inverse_inner_product(restrict(grid, i), inv_closure_stencils, inv_inner_stencil))
    end

    return foldl(⊗, H⁻¹s)
end

function inverse_inner_product(grid::EquidistantGrid{1}, inv_closure_stencils, inv_inner_stencil = CenteredStencil(one(eltype(grid))))
    h⁻¹ = inverse_spacing(grid)
    H⁻¹ = SbpOperators.volume_operator(grid, scale(inv_inner_stencil, h⁻¹[1]), scale.(inv_closure_stencils, h⁻¹[1]),even,1)
    return H⁻¹
end
export inverse_inner_product

inverse_inner_product(grid::EquidistantGrid{0}, inv_closure_stencils, inv_inner_stencil = CenteredStencil(one(eltype(grid)))) = IdentityMapping{eltype(grid)}()

reciprocal_stencil(s::Stencil{T}) where T = Stencil(s.range,one(T)./s.weights)
