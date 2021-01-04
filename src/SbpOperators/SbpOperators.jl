module SbpOperators

using Sbplib.RegionIndices
using Sbplib.LazyTensors
using Sbplib.Grids

include("stencil.jl")
include("d2.jl")
include("readoperator.jl")
include("volumeops/volume_operator.jl")
include("volumeops/derivatives/secondderivative.jl")
include("volumeops/laplace/laplace.jl")
include("volumeops/quadratures/quadrature.jl")
include("volumeops/quadratures/inverse_quadrature.jl")
include("boundaryops/boundary_operator.jl")
include("boundaryops/boundary_restriction.jl")
include("boundaryops/normal_derivative.jl")

end # module
