module SbpOperators

# Stencil set
export StencilSet
export read_stencil_set
export get_stencil_set
export parse_stencil
export parse_nested_stencil
export parse_scalar
export parse_tuple
export sbp_operators_path

# Operators
export boundary_restriction
export inner_product
export inverse_inner_product
export Laplace
export laplace
export normal_derivative
export first_derivative
export second_derivative
export second_derivative_variable
export undivided_skewed04
export closure_size

@enum Parity begin
    odd = -1
    even = 1
end


# Boundary conditions
export BoundaryCondition
export NeumannCondition
export DirichletCondition
export discretize_data
export boundary_data
export boundary
export sat
export sat_tensors

# Using
using Sbplib.RegionIndices
using Sbplib.LazyTensors
using Sbplib.Grids
using LinearAlgebra

# Includes
include("stencil.jl")
include("stencil_set.jl")
include("boundary_conditions/boundary_condition.jl")
include("boundary_conditions/sat.jl")
include("volumeops/volume_operator.jl")
include("volumeops/stencil_operator_distinct_closures.jl")
include("volumeops/constant_interior_scaling_operator.jl")
include("volumeops/derivatives/first_derivative.jl")
include("volumeops/derivatives/second_derivative.jl")
include("volumeops/derivatives/second_derivative_variable.jl")
include("volumeops/derivatives/dissipation.jl")
include("volumeops/laplace/laplace.jl")
include("volumeops/inner_products/inner_product.jl")
include("volumeops/inner_products/inverse_inner_product.jl")
include("boundaryops/boundary_operator.jl")
include("boundaryops/boundary_restriction.jl")
include("boundaryops/normal_derivative.jl")

end # module
