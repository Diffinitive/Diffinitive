"""
    boundary_restriction(grid,closureStencil,boundary)

Creates a BoundaryRestriction operator for the specified boundary
"""
function boundary_restriction(grid::EquidistantGrid{1}, closureStencil::Stencil, boundary::CartesianBoundary{1})
    return e = BoundaryRestriction(grid, closureStencil, region(boundary))
end

function boundary_restriction(grid::EquidistantGrid{2,T}, closureStencil::Stencil{T}, boundary::CartesianBoundary{1}) where T
    e = BoundaryRestriction(restrict(grid, 1), closureStencil, region(boundary))
    I = IdentityMapping{T}(size(restrict(grid,2)))
    return e⊗I
end

function boundary_restriction(grid::EquidistantGrid{2,T}, closureStencil::Stencil{T}, boundary::CartesianBoundary{2}) where T
    e = BoundaryRestriction(restrict(grid, 2), closureStencil, region(boundary))
    I = IdentityMapping{T}(size(restrict(grid,1)))
    return I⊗e
end
export boundary_restriction

"""
    BoundaryRestriction{T,R,N} <: TensorMapping{T,0,1}

Implements the boundary operator `e` for 1D as a TensorMapping
`e` is the restriction of a grid function to the boundary using some `closureStencil`.
The boundary to restrict to is determined by `R`.

`e'` is the prolongation of a zero dimensional array to the whole grid using the same `closureStencil`.
"""
struct BoundaryRestriction{T,R<:Region,N} <: TensorMapping{T,0,1}
    stencil::Stencil{T,N}
    size::Int
end
export BoundaryRestriction

function BoundaryRestriction(grid::EquidistantGrid{1}, closureStencil::Stencil{T,N}, region::Region) where {T,N}
    return BoundaryRestriction{T,typeof(region),N}(closureStencil,size(grid)[1])
end

closuresize(::BoundaryRestriction{T,R,N}) where {T,R,N} = N

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

# Catch all combinations of Lower, Upper and Inner not caught by the two previous methods.
function LazyTensors.apply_transpose(e::BoundaryRestriction{T}, v::AbstractArray{T,0}, i::Index) where T
    return zero(T)
end

function LazyTensors.apply_transpose(e::BoundaryRestriction{T}, v::AbstractArray{T,0}, i) where T
    r = getregion(i, closuresize(e), e.size)
    apply_transpose(e, v, Index(i,r))
end
