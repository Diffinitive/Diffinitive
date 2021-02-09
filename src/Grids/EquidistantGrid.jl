"""
    EquidistantGrid(size::NTuple{Dim, Int}, limit_lower::NTuple{Dim, T}, limit_upper::NTuple{Dim, T})
	EquidistantGrid{T}()

`EquidistantGrid` is a grid with equidistant grid spacing per coordinat direction.

`EquidistantGrid(size, limit_lower, limit_upper)` construct the grid with the
domain defined by the two points P1, and P2 given by `limit_lower` and
`limit_upper`. The length of the domain sides are given by the components of
(P2-P1). E.g for a 2D grid with P1=(-1,0) and P2=(1,2) the domain is defined
as (-1,1)x(0,2). The side lengths of the grid are not allowed to be negative.
The number of equidistantly spaced points in each coordinate direction are given
by `size`.

`EquidistantGrid{T}()` constructs a 0-dimensional grid.

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

	# Specialized constructor for 0-dimensional grid
	EquidistantGrid{T}() where T = new{0,T}((),(),())
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


"""
    boundary_grid(grid::EquidistantGrid,id::CartesianBoundary)
	boundary_grid(::EquidistantGrid{1},::CartesianBoundary{1})

Creates the lower-dimensional restriciton of `grid` spanned by the dimensions
orthogonal to the boundary specified by `id`. The boundary grid of a 1-dimensional
grid is a zero-dimensional grid.
"""
function boundary_grid(grid::EquidistantGrid,id::CartesianBoundary)
	dims = collect(1:dimension(grid))
	orth_dims = dims[dims .!= dim(id)]
	if orth_dims == dims
		throw(DomainError("boundary identifier not matching grid"))
	end
    return restrict(grid,orth_dims)
end
export boundary_grid
boundary_grid(::EquidistantGrid{1,T},::CartesianBoundary{1}) where T = EquidistantGrid{T}()
