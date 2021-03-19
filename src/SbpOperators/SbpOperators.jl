module SbpOperators

using Sbplib.RegionIndices
using Sbplib.LazyTensors
using Sbplib.Grids
using Sbplib.StaticDicts

include("stencil.jl")
include("d2.jl")
include("readoperator.jl")
include("volumeops/volume_operator.jl")
include("volumeops/derivatives/secondderivative.jl")
include("volumeops/laplace/laplace.jl")
include("volumeops/inner_products/inner_product.jl")
include("volumeops/inner_products/inverse_inner_product.jl")
include("boundaryops/boundary_operator.jl")
include("boundaryops/boundary_restriction.jl")
include("boundaryops/normal_derivative.jl")

end # module
