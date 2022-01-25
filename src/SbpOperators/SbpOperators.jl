module SbpOperators

using Sbplib.RegionIndices
using Sbplib.LazyTensors
using Sbplib.Grids
using Sbplib.StaticDicts

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


export inner_product
export inverse_inner_product
export boundary_restriction
export normal_derivative
export boundary_quadrature

end # module
