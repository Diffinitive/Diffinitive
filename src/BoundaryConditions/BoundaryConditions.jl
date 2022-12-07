module BoundaryConditions

export BoundaryData
export ConstantBoundaryData
export SpaceDependentBoundaryData
export TimeDependentBoundaryData
export SpaceTimeDependentBoundaryData
export ZeroBoundaryData
export discretize

export BoundaryCondition
export NeumannCondition
export DirichletCondition
export RobinCondition

export data

export sat
export sat_tensors

using Sbplib.Grids
using Sbplib.LazyTensors

include("boundary_condition.jl")
include("sat.jl")

end # module
