struct TensorGrid{T,D,RD,GT<:NTuple{N,Grid} where N} <: Grid{T,D,RD}
    grids::GT

    function TensorGrid(gs...)
        T = eltype(gs[1]) # All gs should have the same T
        D = sum(ndims,gs)
        RD = sum(nrangedims, gs)

        return new{T,D,RD,typeof(gs)}(gs)
    end
end

function Base.size(g::TensorGrid)
    return concatenate_tuples(size.(g.grids)...)
end

function Base.getindex(g::TensorGrid, I...)
    szs = ndims.(g.grids)

    Is = split_tuple(I, szs)
    ps = map((g,I)->SVector(g[I...]), g.grids, Is)

    return vcat(ps...)
end

IndexStyle(::TensorGrid) = IndexCartesian()

function Base.eachindex(g::TensorGrid)
    szs = concatenate_tuples(size.(g.grids)...)
    return CartesianIndices(szs)
end



## Pre refactor code:
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
    boundary_identifiers(::EquidistantGrid)

Returns a tuple containing the boundary identifiers for the grid, stored as
    (CartesianBoundary(1,Lower),
     CartesianBoundary(1,Upper),
     CartesianBoundary(2,Lower),
     ...)
"""
boundary_identifiers(g::EquidistantGrid) = (((ntuple(i->(CartesianBoundary{i,Lower}(),CartesianBoundary{i,Upper}()),ndims(g)))...)...,)


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
