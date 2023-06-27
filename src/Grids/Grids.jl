module Grids

using Sbplib.RegionIndices
using Sbplib.LazyTensors
using StaticArrays

# Grid
export Grid
export coordinate_size
export component_type

export TensorGrid
export ZeroDimGrid

export TensorGridBoundary

export grid_id
export boundary_id

export eval_on
export getcomponent

# BoundaryIdentifier
export BoundaryIdentifier


# EquidistantGrid
export EquidistantGrid
export spacing
export inverse_spacing
export boundary_identifiers
export boundary_grid
export refine
export coarsen
export equidistant_grid
export CartesianBoundary

export CurvilinearGrid

abstract type BoundaryIdentifier end

include("grid.jl")
include("tensor_grid.jl")
include("equidistant_grid.jl")
include("zero_dim_grid.jl")
include("curvilinear_grid.jl")

end # module
