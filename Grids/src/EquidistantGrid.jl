# EquidistantGrid is a grid with equidistant grid spacing per coordinat
# direction. The domain is defined through the two points P1 = x̄₁, P2 = x̄₂
# by the exterior product of the vectors obtained by projecting (x̄₂-x̄₁) onto
# the coordinate directions. E.g for a 2D grid with x̄₁=(-1,0) and x̄₂=(1,2)
# the domain is defined as (-1,1)x(0,2).

export EquidistantGrid

struct EquidistantGrid{Dim,T<:Real} <: AbstractGrid
    size::NTuple{Dim, Int} # First coordinate direction stored first
    limit_lower::NTuple{Dim, T}
    limit_upper::NTuple{Dim, T}
    spacing::NTuple{Dim, T} # Grid spacing
    inverse_spacing::NTuple{Dim, T} # Reciprocal of grid spacing

    # General constructor
    function EquidistantGrid(size::NTuple{Dim, Int}, limit_lower::NTuple{Dim, T}, limit_upper::NTuple{Dim, T}) where Dim where T
        @assert all(size.>0)
        @assert all(limit_upper.-limit_lower .!= 0)
        spacing = abs.(limit_upper.-limit_lower)./(size.-1)
        inverse_spacing = 1.0./spacing
        return new{Dim,T}(size, limit_lower, limit_upper, spacing, inverse_spacing)
    end
end

function Base.eachindex(grid::EquidistantGrid)
    CartesianIndices(grid.size)
end

Base.size(g::EquidistantGrid) = g.size

# Returns the number of dimensions of an EquidistantGrid.
#
# @Input: grid - an EquidistantGrid
# @Return: dimension - The dimension of the grid
function dimension(grid::EquidistantGrid)
    return length(grid.size)
end

# TODO: Keep the below functions or just use properties?
# Returns the reciprocal of the spacing of the grid
#
function inverse_spacing(grid::EquidistantGrid)
    return grid.inverse_spacing
end
export inverse_spacing

# Returns the reciprocal of the spacing of the grid
#
function spacing(grid::EquidistantGrid)
    return grid.spacing
end
export spacing

# Computes the points of an EquidistantGrid as an array of tuples with
# the same dimension as the grid.
#
# @Input: grid - an EquidistantGrid
# @Return: points - the points of the grid.
function points(grid::EquidistantGrid)
    # TODO: Make this return an abstract array?
    indices = Tuple.(CartesianIndices(grid.size))
    h = spacing(grid)
    return broadcast(I -> grid.limit_lower .+ (I.-1).*h, indices)
end

function pointsalongdim(grid::EquidistantGrid, dim::Integer)
    @assert dim<=dimension(grid)
    @assert dim>0
    points = collect(range(grid.limit_lower[dim],stop=grid.limit_upper[dim],length=grid.size[dim]))
end
