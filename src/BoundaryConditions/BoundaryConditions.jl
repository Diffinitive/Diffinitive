module BoundaryConditions

export BoundaryCondition
export BoundaryConditionType
export Neumann
export Dirichlet

using Sbplib.Grids

include("boundary_condition.jl")

end # module
