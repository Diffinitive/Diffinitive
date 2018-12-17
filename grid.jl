module grid

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
    numberOfDimensions::UInt
    numberOfPoints::Vector{UInt}
    limits::Vector{Pair{Real, Real}}
    function EquidistantGrid(nDims, nPoints, lims)
        @assert nDims == size(nPoints)
        return new(nDims, nPoints, lims)
    end
end

function numberOfDimensions(grid::EquidistantGrid)
    return grid.numberOfDimensions
end

function numberOfPoints(grid::EquidistantGrid)
    return grid.numberOfPoints
end

function points(grid::EquidistantGrid)
    points::Matrix{Real,3}(undef, numberOfPoints(grid))
    return points
end

function limitsForDimension(grid::EquidistantGrid, dim::UInt)
    return grid.limits(dim)
end

end
