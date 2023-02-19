module Grids

using Sbplib.RegionIndices
using Sbplib.LazyTensors

# Grid
export Grid
export dims
export points
export evalOn

# BoundaryIdentifier
export BoundaryIdentifier
export CartesianBoundary
export dim
export region

# EquidistantGrid
export EquidistantGrid
export spacing
export inverse_spacing
export restrict
export boundary_identifiers
export boundary_grid
export refine
export coarsen

include("grid.jl")
include("boundary_identifier.jl")
include("equidistant_grid.jl")
include("zero_dim_grid.jl")

end # module
