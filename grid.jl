module grid
using Plots

abstract type Grid end

function numberOfDimensions(grid::Grid)
    error("Not yet implemented")
end

function numberOfPoints(grid::Grid)
    error("Not yet implemented")
end

function points(grid::Grid)
    error("Not yet implemented")
end

abstract type BoundaryId end

# Move to seperate file.
struct EquidistantGrid <: Grid
    nPointsPerDim::Vector{Int}
    limits::Vector{Pair{Real, Real}}
    function EquidistantGrid(nPointsPerDim, lims)
        @assert length(lims) == length(nPointsPerDim)
        return new(nPointsPerDim, lims)
    end
end

function numberOfDimensions(grid::EquidistantGrid)
    return length(grid.nPointsPerDim)
end

function numberOfPoints(grid::EquidistantGrid)
    numberOfPoints = grid.nPointsPerDim[1];
    for i = 2:length(grid.nPointsPerDim);
        numberOfPoints = numberOfPoints*grid.nPointsPerDim[i]
    end
    return numberOfPoints
end

function points(grid::EquidistantGrid)
    points = Vector{Real}(undef, numberOfPoints(grid))
    for i = 1:numberOfDimensions(grid)
        lims = limitsForDimension(grid,i)
        points = range(lims.first, stop=lims.second, length=grid.nPointsPerDim[i])
    end
    return points
end

function limitsForDimension(grid::EquidistantGrid, dim::Int)
    return grid.limits[dim]
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
