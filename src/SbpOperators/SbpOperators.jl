module SbpOperators

using Sbplib.RegionIndices
using Sbplib.LazyTensors
using Sbplib.Grids

@enum Parity begin
    odd = -1
    even = 1
end

include("stencil.jl")
include("readoperator.jl")
include("volumeops/volume_operator.jl")
include("volumeops/constant_interior_scaling_operator.jl")
include("volumeops/derivatives/second_derivative.jl")
include("volumeops/laplace/laplace.jl")
include("volumeops/inner_products/inner_product.jl")
include("volumeops/inner_products/inverse_inner_product.jl")
include("boundaryops/boundary_operator.jl")
include("boundaryops/boundary_restriction.jl")
include("boundaryops/normal_derivative.jl")


export boundary_quadrature
export boundary_restriction
export inner_product
export inverse_inner_product
export Laplace
export laplace
export normal_derivative
export second_derivative
# REVIEW: Should the exports be placed here or at the top of each file where the thing is defined?
# We currently have a mix of placing it at the place of definition and at the top of each file. This is a "new" way of doing it.

end # module
