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
    return LazyTensors.concatenate_tuples(size.(g.grids)...)
end

function Base.getindex(g::TensorGrid, I...)
    szs = ndims.(g.grids)

    Is = LazyTensors.split_tuple(I, szs)
    ps = map((g,I)->SVector(g[I...]), g.grids, Is)

    return vcat(ps...)
end

IndexStyle(::TensorGrid) = IndexCartesian()

function Base.eachindex(g::TensorGrid)
    szs = LazyTensors.concatenate_tuples(size.(g.grids)...)
    return CartesianIndices(szs)
end


struct TensorBoundary{N, BID<:BoundaryIdentifier} <: BoundaryIdentifier end
grid_id(::TensorBoundary{N, BID}) where {N, BID} = N
boundary_id(::TensorBoundary{N, BID}) where {N, BID} = BID()


"""
    boundary_identifiers(::TensorGrid)

Returns a tuple containing the boundary identifiers for the grid.
"""
function boundary_identifiers(g::TensorGrid)
    n = length(g.grids)
    per_grid = map(eachindex(g.grids)) do i
        return map(bid -> TensorBoundary{i, bid}(), boundary_identifiers(g.grids[i]))
    end
    return LazyTensors.concatenate_tuples(per_grid...)
end


"""
    boundary_grid(grid::TensorGrid, id::TensorBoundary)

The grid for the boundary specified by `id`.
"""
function boundary_grid(g::TensorGrid, bid::TensorBoundary)
    local_boundary_grid = boundary_grid(g.grids[grid_id(bid)], boundary_id(bid))
    new_grids = Base.setindex(g.grids, local_boundary_grid, grid_id(bid))
    return TensorGrid(new_grids...)
end
