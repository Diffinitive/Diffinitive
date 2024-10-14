module Diffinitive

include("RegionIndices/RegionIndices.jl")
include("LazyTensors/LazyTensors.jl")
include("Grids/Grids.jl")
include("SbpOperators/SbpOperators.jl")

export RegionIndices
export LazyTensors
export Grids
export SbpOperators



# Aqua.jl fixes
using StaticArrays
using .LazyTensors
Base.:+(a::StaticArray, b::LazyArray) = a +̃ b
Base.:+(a::LazyArray, b::StaticArray) = a +̃ b
Base.:-(a::StaticArray, b::LazyArray) = a -̃ b
Base.:-(a::LazyArray, b::StaticArray) = a -̃ b

end
