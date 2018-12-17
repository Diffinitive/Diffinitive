module grid

abstract type Grid end

function numberOfDimensions(grid::Grid)
    error("Not yet implemented");
end

function numberOfPoints(grid::Grid)
    error("Not yet implemented");
end

function points(grid::Grid)
    error("Not yet implemented");
end

abstract type BoundaryId end

# Move to seperate file.
struct EquidistantGrid <: Grid
    numberOfDimensions::UInt;
    numberOfPoints::Vector(UInt);
    limits::Vector{Pair{Real, Real}};
    function EquidistantGrid(nDims, nPoints, lims)
        @assert nDims == size(nPoints);
        return new(nDims, nPoints, lims);
    end
end

# Getter functions for public properties?
function numberOfDimensions(grid::EquidistantGrid)
    return grid.numberOfDimensions;
end

function numberOfPoints(grid::EquidistantGrid)
    return grid.numberOfPoints;
end


function points(grid::EquidistantGrid)
    points::Array{Real,3}(undef, numberOfPoints(grid));
#    for i ∈ eachindex(points)
#        points(i) = i/
#    end
    return points;
end

function limitsForDimension(grid::EquidistantGrid, dim::UInt)
    @assert dim <= 3
    @assert dim >= 1
    return grid.limits(dim);
end
