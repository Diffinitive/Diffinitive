module Grid

abstract type BoundaryIdentifier end
struct CartesianBoundary{Dim, R<:Region} <: BoundaryIdentifier end

include("AbstractGrid.jl")
include("EquidistantGrid.jl")

end
