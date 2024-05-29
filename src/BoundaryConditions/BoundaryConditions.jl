module BoundaryConditions

# REVIEW: Does this need to be in a separate module? I feel like it fits quite well into SbpOperators.

export BoundaryCondition
export discretize_data
export data # REVIEW: Name too generic to export.
export id # REVIEW: Also to generic. Change to `boundary`?

export NeumannCondition
export DirichletCondition

export sat
export sat_tensors

using Sbplib.Grids
using Sbplib.LazyTensors

include("boundary_condition.jl")
include("sat.jl")

end # module
