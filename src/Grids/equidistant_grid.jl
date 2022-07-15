
"""
    EquidistantGrid{Dim,T<:Real} <: Grid

`Dim`-dimensional equidistant grid with coordinates of type `T`.
"""
struct EquidistantGrid{Dim,T<:Real} <: Grid
    size::NTuple{Dim, Int}
    limit_lower::NTuple{Dim, T}
    limit_upper::NTuple{Dim, T}

    function EquidistantGrid{Dim,T}(size::NTuple{Dim, Int}, limit_lower::NTuple{Dim, T}, limit_upper::NTuple{Dim, T}) where {Dim,T}
        if any(size .<= 0)
            throw(DomainError("all components of size must be postive"))
        end
        if any(limit_upper.-limit_lower .<= 0)
            throw(DomainError("all side lengths must be postive"))
        end
        return new{Dim,T}(size, limit_lower, limit_upper)
    end
end


"""
    EquidistantGrid(size, limit_lower, limit_upper)

Construct an equidistant grid with corners at the coordinates `limit_lower` and
`limit_upper`.

The length of the domain sides are given by the components of
`limit_upper-limit_lower`. E.g for a 2D grid with `limit_lower=(-1,0)` and `limit_upper=(1,2)` the domain is defined
as `(-1,1)x(0,2)`. The side lengths of the grid are not allowed to be negative.

The number of equidistantly spaced points in each coordinate direction are given
by the tuple `size`.
"""
function EquidistantGrid(size, limit_lower, limit_upper)
    return EquidistantGrid{length(size), eltype(limit_lower)}(size, limit_lower, limit_upper)
end


"""
    EquidistantGrid{T}()

Constructs a 0-dimensional grid.
"""
EquidistantGrid{T}() where T = EquidistantGrid{0,T}((),(),()) # Convenience constructor for 0-dim grid


"""
    EquidistantGrid(size::Int, limit_lower::T, limit_upper::T)

Convenience constructor for 1D grids.
"""
function EquidistantGrid(size::Int, limit_lower::T, limit_upper::T) where T
	return EquidistantGrid((size,),(limit_lower,),(limit_upper,))
end

Base.eltype(grid::EquidistantGrid{Dim,T}) where {Dim,T} = T

Base.eachindex(grid::EquidistantGrid) = CartesianIndices(grid.size)

Base.size(g::EquidistantGrid) = g.size


"""
    dim(grid::EquidistantGrid)

The dimension of the grid.
"""
dim(::EquidistantGrid{Dim}) where Dim = Dim


"""
    spacing(grid::EquidistantGrid)

The spacing between grid points.
"""
spacing(grid::EquidistantGrid) = (grid.limit_upper.-grid.limit_lower)./(grid.size.-1)


"""
    inverse_spacing(grid::EquidistantGrid)

The reciprocal of the spacing between grid points.
"""
inverse_spacing(grid::EquidistantGrid) = 1 ./ spacing(grid)


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

Pick out given dimensions from the grid and return a grid for them.
"""
function restrict(grid::EquidistantGrid, dim)
    size = grid.size[dim]
    limit_lower = grid.limit_lower[dim]
    limit_upper = grid.limit_upper[dim]

    return EquidistantGrid(size, limit_lower, limit_upper)
end


"""
    orthogonal_dims(grid::EquidistantGrid,dim)

Returns the dimensions of grid orthogonal to that of dim.
"""
function orthogonal_dims(grid::EquidistantGrid, dim)
    orth_dims = filter(i -> i != dim, dims(grid))
	if orth_dims == dims(grid)
		throw(DomainError(string("dimension ",string(dim)," not matching grid")))
	end
    return orth_dims
end


"""
    boundary_identifiers(::EquidistantGrid)

Returns a tuple containing the boundary identifiers for the grid, stored as
	(CartesianBoundary(1,Lower),
	 CartesianBoundary(1,Upper),
	 CartesianBoundary(2,Lower),
	 ...)
"""
boundary_identifiers(g::EquidistantGrid) = (((ntuple(i->(CartesianBoundary{i,Lower}(),CartesianBoundary{i,Upper}()),dim(g)))...)...,)


"""
    boundary_grid(grid::EquidistantGrid, id::CartesianBoundary)

Creates the lower-dimensional restriciton of `grid` spanned by the dimensions
orthogonal to the boundary specified by `id`. The boundary grid of a 1-dimensional
grid is a zero-dimensional grid.
"""
function boundary_grid(grid::EquidistantGrid, id::CartesianBoundary)
    orth_dims = orthogonal_dims(grid, dim(id))
    return restrict(grid, orth_dims)
end
boundary_grid(::EquidistantGrid{1,T},::CartesianBoundary{1}) where T = EquidistantGrid{T}()


"""
    boundary_size(grid::EquidistantGrid, id::CartesianBoundary)

Returns the size of the boundary of `grid` specified by `id`.
"""
function boundary_size(grid::EquidistantGrid, id::CartesianBoundary)
	orth_dims = orthogonal_dims(grid, dim(id))
    return  grid.size[orth_dims]
end
boundary_size(::EquidistantGrid{1,T},::CartesianBoundary{1}) where T = ()


"""
    refine(grid::EquidistantGrid, r::Int)

Refines `grid` by a factor `r`. The factor is applied to the number of
intervals which is 1 less than the size of the grid.

See also: [`coarsen`](@ref)
"""
function refine(grid::EquidistantGrid, r::Int)
    sz = size(grid)
    new_sz = (sz .- 1).*r .+ 1
    return EquidistantGrid{dim(grid), eltype(grid)}(new_sz, grid.limit_lower, grid.limit_upper)
end


"""
    coarsen(grid::EquidistantGrid, r::Int)

Coarsens `grid` by a factor `r`. The factor is applied to the number of
intervals which is 1 less than the size of the grid. If the number of
intervals are not divisible by `r` an error is raised.

See also: [`refine`](@ref)
"""
function coarsen(grid::EquidistantGrid, r::Int)
    sz = size(grid)

    if !all(n -> (n % r == 0), sz.-1)
        throw(DomainError(r, "Size minus 1 must be divisible by the ratio."))
    end

    new_sz = (sz .- 1).÷r .+ 1

    return EquidistantGrid{dim(grid), eltype(grid)}(new_sz, grid.limit_lower, grid.limit_upper)
end
