module Grid

abstract type BoundaryIdentifier end
struct CartesianBoundary{Dim, R<:Region} <: BoundaryIdentifier end
dim(::CartesianBoundary{Dim, R}) where {Dim, R} = Dim
region(::CartesianBoundary{Dim, R}) where {Dim, R} = R

include("AbstractGrid.jl")
include("EquidistantGrid.jl")

end
