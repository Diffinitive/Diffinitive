module BoundaryConditions


export BoundaryCondition
export discretize_data
export data
export id

export NeumannCondition
export DirichletCondition

export sat
export sat_tensors

using Sbplib.Grids
using Sbplib.LazyTensors

include("boundary_condition.jl")
include("sat.jl")

end # module
