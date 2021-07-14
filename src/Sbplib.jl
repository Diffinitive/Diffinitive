module Sbplib

include("StaticDicts/StaticDicts.jl")
include("RegionIndices/RegionIndices.jl")
include("LazyTensors/LazyTensors.jl")
include("Grids/Grids.jl")
include("SbpOperators/SbpOperators.jl")
include("DiffOps/DiffOps.jl")

export RegionIndices
export LazyTensors
export Grids
export SbpOperators
export DiffOps

end
