"""
    boundary_restriction(grid,closureStencil,boundary)

Creates a BoundaryRestriction operator for the specified boundary
"""
function boundary_restriction(grid::EquidistantGrid{1,T}, closureStencil::Stencil{T,M}, boundary::CartesianBoundary{1}) where {T,M}
    r = region(boundary)
    return e = BoundaryRestriction(grid, closureStencil, r())
end

function boundary_restriction(grid::EquidistantGrid{2,T}, closureStencil::Stencil{T,M}, boundary::CartesianBoundary{1}) where {T,M}
    r = region(boundary)
    e = BoundaryRestriction(restrict(grid, 1), closureStencil, r())
    I = IdentityMapping{T}(size(restrict(grid,2)))
    return e⊗I
end

function boundary_restriction(grid::EquidistantGrid{2,T}, closureStencil::Stencil{T,M}, boundary::CartesianBoundary{2}) where {T,M}
    r = region(boundary)
    e = BoundaryRestriction(restrict(grid, 2), closureStencil, r())
    I = IdentityMapping{T}(size(restrict(grid,1)))
    return I⊗e
end
export boundary_restriction

"""
    BoundaryRestriction{T,N,R} <: TensorMapping{T,0,1}

Implements the boundary operator `e` as a TensorMapping
"""
struct BoundaryRestriction{T,M,R<:Region} <: TensorMapping{T,0,1}
    stencil::Stencil{T,M}
    size::NTuple{1,Int}
end
export BoundaryRestriction

function BoundaryRestriction(grid::EquidistantGrid{1,T}, closureStencil::Stencil{T,M}, region::Region) where {T,M,R}
    return BoundaryRestriction{T,M,typeof(region)}(closureStencil,size(grid))
end

LazyTensors.range_size(e::BoundaryRestriction) = ()
LazyTensors.domain_size(e::BoundaryRestriction) = e.size

# TODO: Should we support indexing into the 0-dimensional lazyarray? This is
# supported for arrays with linear index style (i.e for e.g
# u = fill(1), u[] and u[1] are both valid.) This currently not supported by
# LazyTensorMappingApplication.
" Restricts a grid function v on a grid of size m to the scalar element v[1]"
function LazyTensors.apply(e::BoundaryRestriction{T,M,Lower}, v::AbstractVector{T}) where {T,M}
    apply_stencil(e.stencil,v,1)
end

" Restricts a grid function v on a grid of size m to the scalar element v[m]"
function LazyTensors.apply(e::BoundaryRestriction{T,M,Upper}, v::AbstractVector{T}) where {T,M}
    apply_stencil_backwards(e.stencil,v,e.size[1])
end

" Transpose of a restriction is an inflation or prolongation.
  Inflates the scalar (1-element) vector to a vector of size of the grid"
function LazyTensors.apply_transpose(e::BoundaryRestriction{T,M,Lower}, v::AbstractArray{T,0}, i) where {T,M}
    @boundscheck if !(0 < Int(i) <= e.size[1])
        throw(BoundsError())
    end
    return e.stencil[Int(i)-1]*v[]
end

" Transpose of a restriction is an inflation or prolongation.
  Inflates the scalar (1-element) vector to a vector of size of the grid"
function LazyTensors.apply_transpose(e::BoundaryRestriction{T,M,Upper}, v::AbstractArray{T,0}, i) where {T,M}
    @boundscheck if !(0 < Int(i) <= e.size[1])
        throw(BoundsError())
    end
    return e.stencil[e.size[1] - Int(i)]*v[]
end
