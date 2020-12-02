module SbpOperators

using Sbplib.RegionIndices
using Sbplib.LazyTensors
using Sbplib.Grids

include("stencil.jl")
include("constantstenciloperator.jl")
include("d2.jl")
include("readoperator.jl")
include("laplace/secondderivative.jl")
include("laplace/laplace.jl")
include("quadrature/diagonal_inner_product.jl")
include("quadrature/quadrature.jl")
include("quadrature/inverse_diagonal_inner_product.jl")
include("quadrature/inverse_quadrature.jl")
include("boundaryops/boundary_restriction.jl")

end # module
