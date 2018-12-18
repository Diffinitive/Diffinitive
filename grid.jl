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

# TODO: Should this be here?
abstract type BoundaryId end

# TODO: Move to seperate file.
# Prefer to use UInt here, but printing UInt returns hex.
struct EquidistantGrid <: Grid
    numberOfPointsPerDim::Tuple
    limits::NTuple{2,Tuple} # Stores the points at the lower and upper corner of the domain.
                            # e.g (-1,0) and (1,2) for a domain of size (-1,1)x(0,2)

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
    # 1D constructor which can be called as EquidistantGrid(m, (x_l,x_r))
    function EquidistantGrid(nPointsPerDim::Int, lims::NTuple{2,Int})
        return EquidistantGrid((nPointsPerDim,), ((lims[1],),(lims[2],)))
    end

end

function numberOfDimensions(grid::EquidistantGrid)
    return length(grid.numberOfPointsPerDim)
end

function numberOfPoints(grid::EquidistantGrid)
    numberOfPoints = grid.numberOfPointsPerDim[1];
    for i = 2:length(grid.numberOfPointsPerDim);
        numberOfPoints = numberOfPoints*grid.numberOfPointsPerDim[i]
    end
    return numberOfPoints
end

# TODO: Decide if spacings should be positive or if it is allowed to be negative
#       If defined as positive, then need to do something extra when calculating the
#       points. The current implementation works for arbitarily given limits of the grid.
function spacings(grid::EquidistantGrid)
    h = Vector{Real}(undef, numberOfDimensions(grid))
    for i ∈ eachindex(h)
        h[i] = (grid.limits[2][i]-grid.limits[1][i])/(grid.numberOfPointsPerDim[i]-1)
    end
    return Tuple(h)
end

function points(grid::EquidistantGrid)
    nPoints = numberOfPoints(grid)
    points = Vector{NTuple{numberOfDimensions(grid),Real}}(undef, nPoints)
    cartesianIndices = CartesianIndices(grid.numberOfPointsPerDim)
    for i ∈ 1:nPoints
        ci = Tuple(cartesianIndices[i]) .-1
        points[i] = grid.limits[1] .+ spacings(grid).*ci
    end
    return points
end

end
