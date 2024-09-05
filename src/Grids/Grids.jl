module Grids

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
export min_spacing
export coarsen
export refine
export eval_on
export componentview
export ArrayComponentView

export BoundaryIdentifier
export TensorGridBoundary
export CartesianBoundary
export LowerBoundary
export UpperBoundary

export TensorGrid
export ZeroDimGrid

export EquidistantGrid
export inverse_spacing
export spacing
export equidistant_grid

include("grid.jl")
include("tensor_grid.jl")
include("equidistant_grid.jl")
include("zero_dim_grid.jl")

end # module
