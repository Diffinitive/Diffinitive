module SbpOperators

using Sbplib.RegionIndices
using Sbplib.LazyTensors

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

abstract type SbpOperator{T,R,D} <: TensorMapping{T,R,D} end

"""
    grid(::ColocationOperator)

Return the the grid which the sbp-operator lives on
"""
function grid end

abstract type ColocationOperator{T,R,D} <: SbpOperator{T,R,D} end

LazyTensors.range_size(co::ColocationOperator) = size(grid(co))
LazyTensors.domain_size(co::ColocationOperator) = size(grid(co))

# Allt ovan kanske är overkill.. Eventuellt bara lättare och tydligare att alla typer definerar sina range och domain size hur dom vill. (I praktiken typ alltid genom att ha gridden som ett fält i structen.)

end # module
