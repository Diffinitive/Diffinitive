# TODO: Double check that the interfaces for indexing and iterating are fully implemented and tested for all grids.
module Grids

using Sbplib.RegionIndices
using Sbplib.LazyTensors
using StaticArrays


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

export Chart
export ConcreteChart
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
export coarsen
export refine
export eval_on
export componentview
export ArrayComponentView

export BoundaryIdentifier
export TensorGridBoundary
export CartesianBoundary

export TensorGrid
export ZeroDimGrid

export EquidistantGrid
export inverse_spacing
export spacing
export equidistant_grid


# MappedGrid
export MappedGrid
export jacobian
export logicalgrid
export mapped_grid
export jacobian_determinant
export geometric_tensor
export geometric_tensor_inverse

abstract type BoundaryIdentifier end

include("manifolds.jl")
include("grid.jl")
include("tensor_grid.jl")
include("equidistant_grid.jl")
include("zero_dim_grid.jl")
include("mapped_grid.jl")

end # module
