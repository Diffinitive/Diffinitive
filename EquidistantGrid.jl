# EquidistantGrid is a grid with equidistant grid spacing per coordinat
# direction. The domain is defined through the two points P1 = x̄₁, P2 = x̄₂
# by the exterior product of the vectors obtained by projecting (x̄₂-x̄₁) onto
# the coordinate directions. E.g for a 2D grid with x̄₁=(-1,0) and x̄₂=(1,2)
# the domain is defined as (-1,1)x(0,2).

struct EquidistantGrid{Dim,T<:Real} <: AbstractGrid
    size::NTuple{Dim, Int} # First coordinate direction stored first
    limit_lower::NTuple{Dim, T}
    limit_upper::NTuple{Dim, T}
    spacing::NTuple{Dim, T}

    # General constructor
    function EquidistantGrid(size::NTuple{Dim, Int}, limit_lower::NTuple{Dim, T}, limit_upper::NTuple{Dim, T}) where Dim where T
        @assert all(size.>0)
        @assert all(limit_upper.-limit_lower .!= 0)
        spacing = abs.(limit_upper.-limit_lower)./(size.-1)
        return new{Dim,T}(size, limit_lower, limit_upper, spacing)
    end
end

# Returns the number of dimensions of an EquidistantGrid.
#
# @Input: grid - an EquidistantGrid
# @Return: dimension - The dimension of the grid
function dimension(grid::EquidistantGrid)
    return length(grid.size)
end

function Base.eachindex(grid::EquidistantGrid)
    CartesianIndices(grid.size)
end

# Computes the points of an EquidistantGrid as a vector of tuples. The vector is ordered
# such that points in the first coordinate direction varies first, then the second
# and lastely the third (if applicable)
#
# @Input: grid - an EquidistantGrid
# @Return: points - the points of the grid.
function points(grid::EquidistantGrid)
    # TODO: Make this return an abstract array?
    physical_domain_size = (grid.limit_upper .- grid.limit_lower)
    indices = Tuple.(CartesianIndices(grid.size))
    return broadcast(I -> grid.limit_lower .+ physical_domain_size.*(I.-1), indices)
end

function pointsalongdim(grid::EquidistantGrid, dim::Integer)
    @assert dim<=dimension(grid)
    @assert dim>0
    points = range(grid.limit_lower[dim],stop=grid.limit_lower[dim],length=grid.size[dim])
end
