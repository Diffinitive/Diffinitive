module Grids

using Diffinitive.LazyTensors
using StaticArrays
using LinearAlgebra

export ParameterSpace
export HyperBox
export Simplex
export Interval
export Rectangle
export Box
export Triangle
export Tetrahedron

export limits
export unitinterval
export unitsquare
export unitcube
export unithyperbox

export verticies
export unittriangle
export unittetrahedron
export unitsimplex

export Chart

export Atlas
export charts
export connections
export boundaries
export CartesianAtlas

export parameterspace

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
export normal

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

export MultiBlockBoundary


# MappedGrid
export MappedGrid
export jacobian
export logical_grid
export mapped_grid
export metric_tensor

include("parameter_space.jl")
include("grid.jl")
include("multiblockgrids.jl")
include("manifolds.jl")
include("tensor_grid.jl")
include("equidistant_grid.jl")
include("zero_dim_grid.jl")
include("mapped_grid.jl")

end # module
