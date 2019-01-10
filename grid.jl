module grid
using Plots

abstract type Grid end

function numberOfDimensions(grid::Grid)
    error("Not implemented for abstact type Grid")
end

function numberOfPoints(grid::Grid)
    error("Not implemented for abstact type Grid")
end

function points(grid::Grid)
    error("Not implemented for abstact type Grid")
end

# TODO: Should this be here?
abstract type BoundaryId end

# EquidistantGrid is a grid with equidisant grid spacing per coordinat
# direction. The domain is defined through the two points P1 = x̄₁, P2 = x̄₂
# by the exterior product of the vectors obtained by projecting (x̄₂-x̄₁) onto
# the coordinate directions. E.g for a 2D grid with x̄₁=(-1,0) and x̄₂=(1,2)
# the domain is defined as (-1,1)x(0,2).
struct EquidistantGrid <: Grid
    numberOfPointsPerDim::Tuple # First coordinate direction stored first, then
                                # second, then third.
    limits::NTuple{2,Tuple} # Stores the two points which defines the range of
                            # the e.g (-1,0) and (1,2) for a domain of size
                            # (-1,1)x(0,2)

    # General constructor
    function EquidistantGrid(nPointsPerDim::Tuple, lims::NTuple{2,Tuple})
        @assert length(nPointsPerDim) > 0
        @assert count(x -> x > 0, nPointsPerDim) == length(nPointsPerDim)
        @assert length(lims[1]) == length(nPointsPerDim)
        @assert length(lims[2]) == length(nPointsPerDim)
        # TODO: Assert that the same values are not passed in both lims[1] and lims[2]
        #       i.e the domain length is positive for all dimensions
        return new(nPointsPerDim, lims)
    end
    # 1D constructor which can be called as EquidistantGrid(m, (xl,xr))
    function EquidistantGrid(nPointsPerDim::Integer, lims::NTuple{2,Integer})
        return EquidistantGrid((nPointsPerDim,), ((lims[1],),(lims[2],)))
    end

end

# Returns the number of dimensions of an EquidistantGrid.
#
# @Input: grid - an EquidistantGrid
# @Return: numberOfPoints - The number of dimensions
function numberOfDimensions(grid::EquidistantGrid)
    return length(grid.numberOfPointsPerDim)
end

# Computes the total number of points of an EquidistantGrid.
#
# @Input: grid - an EquidistantGrid
# @Return: numberOfPoints - The total number of points
function numberOfPoints(grid::EquidistantGrid)
    numberOfPoints = grid.numberOfPointsPerDim[1];
    for i = 2:length(grid.numberOfPointsPerDim);
        numberOfPoints = numberOfPoints*grid.numberOfPointsPerDim[i]
    end
    return numberOfPoints
end

# Computes the grid spacing of an EquidistantGrid, i.e the unsigned distance
# between two points for each coordinate direction.
#
# @Input: grid - an EquidistantGrid
# @Return: h̄ - Grid spacing for each coordinate direction stored in a tuple.
function spacings(grid::EquidistantGrid)
    h̄ = Vector{Real}(undef, numberOfDimensions(grid))
    for i ∈ eachindex(h̄)
        h̄[i] = abs(grid.limits[2][i]-grid.limits[1][i])/(grid.numberOfPointsPerDim[i]-1)
    end
    return Tuple(h̄)
end

# Computes the points of an EquidistantGrid as a vector of tuples. The vector is ordered
# such that points in the first coordinate direction varies first, then the second
# and lastely the third (if applicable)
#
# @Input: grid - an EquidistantGrid
# @Return: points - the points of the grid.
function points(grid::EquidistantGrid)
    # Compute signed grid spacings
    dx̄ = Vector{Real}(undef, numberOfDimensions(grid))
    for i ∈ eachindex(dx̄)
        dx̄[i] = (grid.limits[2][i]-grid.limits[1][i])/(grid.numberOfPointsPerDim[i]-1)
    end
    dx̄ = Tuple(dx̄)

    nPoints = numberOfPoints(grid)
    points = Vector{NTuple{numberOfDimensions(grid),Real}}(undef, nPoints)
    # Compute the points based on their Cartesian indices and the signed
    # grid spacings
    cartesianIndices = CartesianIndices(grid.numberOfPointsPerDim)
    for i ∈ 1:nPoints
        ci = Tuple(cartesianIndices[i]) .-1
        points[i] = grid.limits[1] .+ dx̄.*ci
    end
    return points
end

function plotOnGrid(grid::EquidistantGrid,v::Vector)
    dim = numberOfDimensions(grid)
    x = points(grid)

    if dim ==1
        plot(x,v)
    else
        error(string("Plot not implemented for dim =", string(dim)))
    end
end

end
