"""
    boundary_restriction(grid,closureStencil,boundary)

Creates a boundary restriction operator on a `Dim`-dimensional grid for the
specified `boundary`.

When `Dim=1`, the corresponding `BoundaryRestriction` tensor mapping is returned.
When `Dim>1`, the `BoundaryRestriction` `e` is inflated by the outer product
of `IdentityMappings` in orthogonal coordinate directions, e.g for `Dim=3`,
the boundary restriction operator in the y-direction direction is `Ix⊗e⊗Iz`.
"""
function boundary_restriction(grid::EquidistantGrid{Dim,T}, closureStencil::Stencil{T,M}, boundary::CartesianBoundary) where {Dim,T,M}
    # Create 1D boundary restriction operator
    r = region(boundary)
    d = dim(boundary)
    e = BoundaryRestriction(restrict(grid, d), closureStencil, r)

    # Create 1D IdentityMappings for each coordinate direction
    one_d_grids = restrict.(Ref(grid), Tuple(1:D))
    Is = IdentityMapping{T}.(size.(one_d_grids))

    # Formulate the correct outer product sequence of the identity mappings and
    # the boundary restriction operator
    parts = Base.setindex(Is, e, d)
    return foldl(⊗, parts)
end

export boundary_restriction

"""
    BoundaryRestriction{T,R,N} <: TensorMapping{T,0,1}

Implements the boundary operator `e` for 1D as a `TensorMapping`

`e` is the restriction of a grid function to the boundary using some `closureStencil`.
The boundary to restrict to is determined by `R`.

`e'` is the prolongation of a zero dimensional array to the whole grid using the same `closureStencil`.
"""
struct BoundaryRestriction{T,R<:Region,N} <: TensorMapping{T,0,1}
    stencil::Stencil{T,N}
    size::Int
end
export BoundaryRestriction

BoundaryRestriction{R}(stencil::Stencil{T,N}, size::Int) where {T,R,N} = BoundaryRestriction{T,R,N}(stencil, size)

function BoundaryRestriction(grid::EquidistantGrid{1}, closureStencil::Stencil{T,N}, region::Region) where {T,N}
    return BoundaryRestriction{T,typeof(region),N}(closureStencil,size(grid)[1])
end

closure_size(::BoundaryRestriction{T,R,N}) where {T,R,N} = N

LazyTensors.range_size(e::BoundaryRestriction) = ()
LazyTensors.domain_size(e::BoundaryRestriction) = (e.size,)

function LazyTensors.apply(e::BoundaryRestriction{T,Lower}, v::AbstractVector{T}) where T
    apply_stencil(e.stencil,v,1)
end

function LazyTensors.apply(e::BoundaryRestriction{T,Upper}, v::AbstractVector{T}) where T
    apply_stencil_backwards(e.stencil,v,e.size)
end

function LazyTensors.apply_transpose(e::BoundaryRestriction{T,Lower}, v::AbstractArray{T,0}, i::Index{Lower}) where T
    return e.stencil[Int(i)-1]*v[]
end

function LazyTensors.apply_transpose(e::BoundaryRestriction{T,Upper}, v::AbstractArray{T,0}, i::Index{Upper}) where T
    return e.stencil[e.size[1] - Int(i)]*v[]
end

# Catch all combinations of Lower, Upper and Interior not caught by the two previous methods.
function LazyTensors.apply_transpose(e::BoundaryRestriction{T}, v::AbstractArray{T,0}, i::Index) where T
    return zero(T)
end

function LazyTensors.apply_transpose(e::BoundaryRestriction{T}, v::AbstractArray{T,0}, i) where T
    r = getregion(i, closure_size(e), e.size)
    apply_transpose(e, v, Index(i,r))
end
