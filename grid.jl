module Grid

# TODO: Where is this used?
abstract type BoundaryId end

include("AbstractGrid.jl")
include("EquidistantGrid.jl")

end
