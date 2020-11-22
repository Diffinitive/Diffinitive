"""
    BoundaryRestriction{T,N,R} <: TensorMapping{T,1,1}

Implements the boundary operator `e` as a TensorMapping
"""
struct BoundaryRestriction{T,M,R<:Region} <: TensorMapping{T,1,1}
    closureStencil::Stencil{T,M}
    size::NTuple{1,Int}
end
export BoundaryRestriction


function BoundaryRestriction(grid::EquidistantGrid{1,T}, closureStencil::Stencil{T,M},region::Region) where {T,M}
    return BoundaryRestriction{T,M,typeof(region)}(closureStencil,size(grid))
end

LazyTensors.range_size(e::BoundaryRestriction) = (1,)
LazyTensors.domain_size(e::BoundaryRestriction) = e.size

" Restricts a grid function v on a grid of size (m,) to the scalar element v[1]"
function LazyTensors.apply(e::BoundaryRestriction{T,M,Lower}, v::AbstractVector{T}, i) where {T,M}
    # TODO: Currently closuresize = 4 for the stencil,
    # but we should only be able to apply this for v[1] since the result is scalar.
    @boundscheck if !(0 < Int(i) <= closuresize(e))
        throw(BoundsError())
    end
    apply_stencil(e.closureStencil,v,Int(i))
end

" Restricts a grid function v on a grid of size (m,) to the scalar element v[m]"
function LazyTensors.apply(e::BoundaryRestriction{T,M,Upper}, v::AbstractVector{T}, i) where {T,M}
    # TODO: Currently closuresize = 4 for the stencil,
    # but we should only be able to apply this for v[1] since the result is scalar.
    @boundscheck if !(e.size[1] - closuresize(e) < Int(i) <= e.size[1])
        throw(BoundsError())
    end
    apply_stencil_backwards(e.closureStencil,v,Int(i))
end

" Transpose of a restriction is an inflation or prolongation.
  Inflates the scalar (1-element) vector to a vector of size of the grid"
function LazyTensors.apply_transpose(e::BoundaryRestriction{T,M,Lower}, v::AbstractVector{T}, i) where {T,M}
    @boundscheck if !(0 < Int(i) <= e.size[1])
        throw(BoundsError())
    end
    return e.closureStencil[Int(i)-1]*v[1]
end

closuresize(e::BoundaryRestriction{T,M}) where {T,M} = M
