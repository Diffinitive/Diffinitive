"""
    EquidistantGrid(size::NTuple{Dim, Int}, limit_lower::NTuple{Dim, T}, limit_upper::NTuple{Dim, T}

EquidistantGrid is a grid with equidistant grid spacing per coordinat direction.
The domain is defined through the two points P1 = x̄₁, P2 = x̄₂ by the exterior
product of the vectors obtained by projecting (x̄₂-x̄₁) onto the coordinate
directions. E.g for a 2D grid with x̄₁=(-1,0) and x̄₂=(1,2) the domain is defined
as (-1,1)x(0,2). The side lengths of the grid are not allowed to be negative
"""
struct EquidistantGrid{Dim,T<:Real} <: AbstractGrid
    size::NTuple{Dim, Int}
    limit_lower::NTuple{Dim, T}
    limit_upper::NTuple{Dim, T}

    # General constructor
    function EquidistantGrid(size::NTuple{Dim, Int}, limit_lower::NTuple{Dim, T}, limit_upper::NTuple{Dim, T}) where Dim where T
        if any(size .<= 0)
            throw(DomainError("all components of size must be postive"))
        end
        if any(limit_upper.-limit_lower .<= 0)
            throw(DomainError("all side lengths must be postive"))
        end
        return new{Dim,T}(size, limit_lower, limit_upper)
    end
end
export EquidistantGrid


"""
    EquidistantGrid(size::Int, limit_lower::T, limit_upper::T)

Convenience constructor for 1D grids.
"""
function EquidistantGrid(size::Int, limit_lower::T, limit_upper::T) where T
	return EquidistantGrid((size,),(limit_lower,),(limit_upper,))
end

function Base.eachindex(grid::EquidistantGrid)
    CartesianIndices(grid.size)
end

Base.size(g::EquidistantGrid) = g.size

"""
    dimension(grid::EquidistantGrid)

The dimension of the grid.
"""
dimension(grid::EquidistantGrid{Dim}) where Dim = Dim

"""
    spacing(grid::EquidistantGrid)

The spacing between the grid points of the grid.
"""
spacing(grid::EquidistantGrid) = (grid.limit_upper.-grid.limit_lower)./(grid.size.-1)
export spacing

"""
    inverse_spacing(grid::EquidistantGrid)

The reciprocal of the spacing between the grid points of the grid.
"""
inverse_spacing(grid::EquidistantGrid) = 1 ./ spacing(grid)
export inverse_spacing

"""
    points(grid::EquidistantGrid)

The point of the grid as an array of tuples with the same dimension as the grid.
The points are stored as [(x1,y1), (x1,y2), … (x1,yn);
						  (x2,y1), (x2,y2), … (x2,yn);
						  	⋮		 ⋮            ⋮
						  (xm,y1), (xm,y2), … (xm,yn)]
"""
function points(grid::EquidistantGrid)
    indices = Tuple.(CartesianIndices(grid.size))
    h = spacing(grid)
    return broadcast(I -> grid.limit_lower .+ (I.-1).*h, indices)
end

"""
    restrict(::EquidistantGrid, dim)

Pick out given dimensions from the grid and return a grid for them
"""
function restrict(grid::EquidistantGrid, dim)
    size = grid.size[dim]
    limit_lower = grid.limit_lower[dim]
    limit_upper = grid.limit_upper[dim]

    return EquidistantGrid(size, limit_lower, limit_upper)
end
export restrict

"""
    boundary_identifiers(::EquidistantGrid)

Returns a tuple containing the boundary identifiers for the grid, stored as
	(CartesianBoundary(1,Lower),
	 CartesianBoundary(1,Upper),
	 CartesianBoundary(2,Lower),
	 ...)
"""
boundary_identifiers(g::EquidistantGrid) = (((ntuple(i->(CartesianBoundary{i,Lower}(),CartesianBoundary{i,Upper}()),dimension(g)))...)...,)
export boundary_identifiers
