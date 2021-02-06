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
`SbpOperators.volume_operator(...)` for more details.
"""
function quadrature(grid::EquidistantGrid, inner_stencil, closure_stencils) where Dim
    h = spacing(grid)
    H = SbpOperators.volume_operator(grid, scale(inner_stencil,h[1]), scale.(closure_stencils,h[1]), even, 1)
    for i ∈ 2:dimension(grid)
        Hᵢ = SbpOperators.volume_operator(grid, scale(inner_stencil,h[i]), scale.(closure_stencils,h[i]), even, i)
        H = H∘Hᵢ
    end
    return H
end
export quadrature

function quadrature(grid::EquidistantGrid, closure_stencils::NTuple{M,Stencil{T}}) where {M,T}
    inner_stencil = Stencil(Tuple{T}(1),center=1)
    return quadrature(grid, inner_stencil, closure_stencils)
end

"""
    boundary_quadrature(grid::EquidistantGrid, inner_stencil, closure_stencils, id::CartesianBoundary)
    boundary_quadrature(grid::EquidistantGrid{1}, inner_stencil, closure_stencils, id)
    boundary_quadrature(grid::EquidistantGrid, closure_stencils, id)

Creates the lower-dimensional quadrature operator associated with the boundary
of `grid` specified by `id`. The quadrature operator is defined on the grid
spanned by the dimensions orthogonal to the boundary coordinate direction.
If the dimension of `grid` is 1, then the boundary quadrature is the 0-dimensional
`IdentityMapping`. If `inner_stencil` is omitted a central interior stencil with
weight 1 is used.
"""
function boundary_quadrature(grid::EquidistantGrid, inner_stencil, closure_stencils, id::CartesianBoundary)
    return quadrature(orthogonal_grid(grid,dim(id)),inner_stencil,closure_stencils)
end
export boundary_quadrature

function boundary_quadrature(grid::EquidistantGrid{1}, inner_stencil::Stencil{T}, closure_stencils::NTuple{M,Stencil{T}}, id::CartesianBoundary{1}) where {M,T}
    return IdentityMapping{T}()
end

function boundary_quadrature(grid::EquidistantGrid, closure_stencils::NTuple{M,Stencil{T}}, id::CartesianBoundary) where {M,T}
    inner_stencil = Stencil(Tuple{T}(1),center=1)
    return boundary_quadrature(grid,inner_stencil,closure_stencils,id)
end

"""
    orthogonal_grid(grid,dim)

Creates the lower-dimensional restriciton of `grid` spanned by the dimensions
orthogonal to `dim`.
"""
function orthogonal_grid(grid,dim)
    dims = collect(1:dimension(grid))
    orth_dims = dims[dims .!= dim]
    return restrict(grid,orth_dims)
end
