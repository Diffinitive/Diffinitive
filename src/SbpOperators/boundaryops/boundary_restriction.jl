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

Implements the boundary operator `e` as a TensorMapping
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

" Restricts a grid function v on a grid of size m to the scalar element v[1]"
function LazyTensors.apply(e::BoundaryRestriction{T,N,Lower}, v::AbstractVector{T}) where {T,N}
    apply_stencil(e.stencil,v,1)
end

" Restricts a grid function v on a grid of size m to the scalar element v[m]"
function LazyTensors.apply(e::BoundaryRestriction{T,N,Upper}, v::AbstractVector{T}) where {T,N}
    apply_stencil_backwards(e.stencil,v,e.size[1])
end

" Transpose of a restriction is an inflation or prolongation.
  Inflates the scalar (1-element) vector to a vector of size of the grid"
function LazyTensors.apply_transpose(e::BoundaryRestriction{T,N,Lower}, v::AbstractArray{T,0}, i) where {T,N}
    @boundscheck if !(0 < Int(i) <= e.size[1])
        throw(BoundsError())
    end
    return e.stencil[Int(i)-1]*v[]
end

" Transpose of a restriction is an inflation or prolongation.
  Inflates the scalar (1-element) vector to a vector of size of the grid"
function LazyTensors.apply_transpose(e::BoundaryRestriction{T,N,Upper}, v::AbstractArray{T,0}, i) where {T,N}
    @boundscheck if !(0 < Int(i) <= e.size[1])
        throw(BoundsError())
    end
    return e.stencil[e.size[1] - Int(i)]*v[]
end
