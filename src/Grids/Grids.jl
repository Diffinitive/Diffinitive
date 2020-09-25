module Grids

using RegionIndices

export BoundaryIdentifier, CartesianBoundary

abstract type BoundaryIdentifier end
struct CartesianBoundary{Dim, R<:Region} <: BoundaryIdentifier end
dim(::CartesianBoundary{Dim, R}) where {Dim, R} = Dim
region(::CartesianBoundary{Dim, R}) where {Dim, R} = R

export dim, region

include("AbstractGrid.jl")
include("EquidistantGrid.jl")

end # module
