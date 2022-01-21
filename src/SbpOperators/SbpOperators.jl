module SbpOperators

using Sbplib.RegionIndices
using Sbplib.LazyTensors
using Sbplib.Grids

@enum Parity begin
    odd = -1
    even = 1
end

export closure_size

include("stencil.jl")
include("readoperator.jl")
include("volumeops/volume_operator.jl")
include("volumeops/constant_interior_scaling_operator.jl")
include("volumeops/derivatives/second_derivative.jl")
include("volumeops/derivatives/second_derivative_variable.jl")
include("volumeops/laplace/laplace.jl")
include("volumeops/inner_products/inner_product.jl")
include("volumeops/inner_products/inverse_inner_product.jl")
include("boundaryops/boundary_operator.jl")
include("boundaryops/boundary_restriction.jl")
include("boundaryops/normal_derivative.jl")

end # module
