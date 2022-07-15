module Grids

using Sbplib.RegionIndices

# Grid
export Grid
export dim
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
export orthogonal_dims
export boundary_identifiers
export boundary_grid
export boundary_size
export refine
export coarsen

include("grid.jl")
include("boundary_identifier.jl")
include("equidistant_grid.jl")

end # module
