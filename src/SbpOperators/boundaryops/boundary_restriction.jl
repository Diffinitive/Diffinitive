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
    BoundaryRestriction{T,N,R} <: TensorMapping{T,0,1}

Implements the boundary operator `e` for 1D as a TensorMapping
`e` is the restriction of a grid function to the boundary using some `closureStencil`.
The boundary to restrict to is determined by `R`.

`e'` is the prolongation of a zero dimensional array to the whole grid using the same `closureStencil`.
"""
struct BoundaryRestriction{T,N,R<:Region} <: TensorMapping{T,0,1}
    stencil::Stencil{T,N}
    size::NTuple{1,Int}
end
export BoundaryRestriction

function BoundaryRestriction(grid::EquidistantGrid{1}, closureStencil::Stencil{T,N}, region::Region) where {T,N}
    return BoundaryRestriction{T,N,typeof(region)}(closureStencil,size(grid))
end

LazyTensors.range_size(e::BoundaryRestriction) = ()
LazyTensors.domain_size(e::BoundaryRestriction) = e.size

function LazyTensors.apply(e::BoundaryRestriction{T,N,Lower}, v::AbstractVector{T}) where {T,N}
    apply_stencil(e.stencil,v,1)
end

function LazyTensors.apply(e::BoundaryRestriction{T,N,Upper}, v::AbstractVector{T}) where {T,N}
    apply_stencil_backwards(e.stencil,v,e.size[1])
end

function LazyTensors.apply_transpose(e::BoundaryRestriction{T,N,Lower}, v::AbstractArray{T,0}, i) where {T,N}
    @boundscheck if !(0 < Int(i) <= e.size[1])
        throw(BoundsError())
    end
    return e.stencil[Int(i)-1]*v[]
end

function LazyTensors.apply_transpose(e::BoundaryRestriction{T,N,Upper}, v::AbstractArray{T,0}, i) where {T,N}
    @boundscheck if !(0 < Int(i) <= e.size[1])
        throw(BoundsError())
    end
    return e.stencil[e.size[1] - Int(i)]*v[]
end
