module Grids

using Sbplib.RegionIndices
using Sbplib.LazyTensors
using StaticArrays

# Grid
export Grid
export coordinate_size
export component_type
export grid_id
export boundary_id
export boundary_indices
export boundary_identifiers
export boundary_grid
export coarsen
export refine
export eval_on
export componentview
export ArrayComponentView
export orthogonal_grid

export BoundaryIdentifier
export TensorGridBoundary
export CartesianBoundary

export TensorGrid
export ZeroDimGrid

export EquidistantGrid
export inverse_spacing
export spacing
export equidistant_grid


abstract type BoundaryIdentifier end

include("grid.jl")
include("tensor_grid.jl")
include("equidistant_grid.jl")
include("zero_dim_grid.jl")

end # module
