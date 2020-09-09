module DiffOps

using RegionIndices
using SbpOperators
using Grids
using LazyTensors

"""
    DiffOp

Supertype of differential operator discretisations.
The action of the DiffOp is defined in the method
    apply(D::DiffOp, v::AbstractVector, I...)
"""
abstract type DiffOp end

function apply end

function matrixRepresentation(D::DiffOp)
    error("not implemented")
end

abstract type DiffOpCartesian{Dim} <: DiffOp end

# DiffOp must have a grid of dimension Dim!!!
function apply!(D::DiffOpCartesian{Dim}, u::AbstractArray{T,Dim}, v::AbstractArray{T,Dim}) where {T,Dim}
    for I ∈ eachindex(D.grid)
        u[I] = apply(D, v, I)
    end

    return nothing
end
export apply!

function apply_region!(D::DiffOpCartesian{2}, u::AbstractArray{T,2}, v::AbstractArray{T,2}) where T
    apply_region!(D, u, v, Lower, Lower)
    apply_region!(D, u, v, Lower, Interior)
    apply_region!(D, u, v, Lower, Upper)
    apply_region!(D, u, v, Interior, Lower)
    apply_region!(D, u, v, Interior, Interior)
    apply_region!(D, u, v, Interior, Upper)
    apply_region!(D, u, v, Upper, Lower)
    apply_region!(D, u, v, Upper, Interior)
    apply_region!(D, u, v, Upper, Upper)
    return nothing
end

# Maybe this should be split according to b3fbef345810 after all?! Seems like it makes performance more predictable
function apply_region!(D::DiffOpCartesian{2}, u::AbstractArray{T,2}, v::AbstractArray{T,2}, r1::Type{<:Region}, r2::Type{<:Region}) where T
    for I ∈ regionindices(D.grid.size, closuresize(D.op), (r1,r2))
        @inbounds indextuple = (Index{r1}(I[1]), Index{r2}(I[2]))
        @inbounds u[I] = apply(D, v, indextuple)
    end
    return nothing
end
export apply_region!

function apply_tiled!(D::DiffOpCartesian{2}, u::AbstractArray{T,2}, v::AbstractArray{T,2}) where T
    apply_region_tiled!(D, u, v, Lower, Lower)
    apply_region_tiled!(D, u, v, Lower, Interior)
    apply_region_tiled!(D, u, v, Lower, Upper)
    apply_region_tiled!(D, u, v, Interior, Lower)
    apply_region_tiled!(D, u, v, Interior, Interior)
    apply_region_tiled!(D, u, v, Interior, Upper)
    apply_region_tiled!(D, u, v, Upper, Lower)
    apply_region_tiled!(D, u, v, Upper, Interior)
    apply_region_tiled!(D, u, v, Upper, Upper)
    return nothing
end

using TiledIteration
function apply_region_tiled!(D::DiffOpCartesian{2}, u::AbstractArray{T,2}, v::AbstractArray{T,2}, r1::Type{<:Region}, r2::Type{<:Region}) where T
    ri = regionindices(D.grid.size, closuresize(D.op), (r1,r2))
    # TODO: Pass Tilesize to function
    for tileaxs ∈ TileIterator(axes(ri), padded_tilesize(T, (5,5), 2))
        for j ∈ tileaxs[2], i ∈ tileaxs[1]
            I = ri[i,j]
            u[I] = apply(D, v, (Index{r1}(I[1]), Index{r2}(I[2])))
        end
    end
    return nothing
end
export apply_region_tiled!

function apply(D::DiffOp, v::AbstractVector)::AbstractVector
    u = zeros(eltype(v), size(v))
    apply!(D,v,u)
    return u
end

export apply

"""
A BoundaryCondition should implement the method
    sat(::DiffOp, v::AbstractArray, data::AbstractArray, ...)
"""
abstract type BoundaryCondition end


end # module
