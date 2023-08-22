"""
    TensorGrid{T,D} <: Grid{T,D}

A grid constructed as the tensor product of other grids.

Currently only supports grids with the `HasShape`-trait.
"""
struct TensorGrid{T,D,GT<:NTuple{N,Grid} where N} <: Grid{T,D}
    grids::GT

    function TensorGrid(gs...)
        T = mapreduce(eltype, combined_coordinate_vector_type, gs)
        D = sum(ndims, gs)

        return new{T,D,typeof(gs)}(gs)
    end
end

# Indexing interface
function Base.getindex(g::TensorGrid, I::Vararg{Int})
    szs = ndims.(g.grids)

    Is = LazyTensors.split_tuple(I, szs)
    ps = map((g,I)->SVector(g[I...]), g.grids, Is)

    return vcat(ps...)
end

function Base.eachindex(g::TensorGrid)
    szs = LazyTensors.concatenate_tuples(size.(g.grids)...)
    return CartesianIndices(szs)
end

# Iteration interface
Base.iterate(g::TensorGrid) = iterate(Iterators.product(g.grids...)) |> _iterate_combine_coords
Base.iterate(g::TensorGrid, state) = iterate(Iterators.product(g.grids...), state) |> _iterate_combine_coords
_iterate_combine_coords(::Nothing) = nothing
_iterate_combine_coords((next,state)) = combine_coordinates(next...), state

Base.IteratorSize(::Type{<:TensorGrid{<:Any, D}}) where D = Base.HasShape{D}()
Base.eltype(::Type{<:TensorGrid{T}}) where T = T
Base.length(g::TensorGrid) = sum(length, g.grids)
Base.size(g::TensorGrid) = LazyTensors.concatenate_tuples(size.(g.grids)...)


refine(g::TensorGrid, r::Int) = mapreduce(g->refine(g,r), TensorGrid, g.grids)
coarsen(g::TensorGrid, r::Int) = mapreduce(g->coarsen(g,r), TensorGrid, g.grids)

"""
    TensorGridBoundary{N, BID} <: BoundaryIdentifier

A boundary identifier for a tensor grid. `N` Specifies which grid in the
tensor product and `BID` which boundary on that grid.
"""
struct TensorGridBoundary{N, BID} <: BoundaryIdentifier end
grid_id(::TensorGridBoundary{N, BID}) where {N, BID} = N
boundary_id(::TensorGridBoundary{N, BID}) where {N, BID} = BID()

"""
    boundary_identifiers(g::TensorGrid)

Returns a tuple containing the boundary identifiers of `g`.
"""
function boundary_identifiers(g::TensorGrid)
    per_grid = map(eachindex(g.grids)) do i
        return map(bid -> TensorGridBoundary{i, typeof(bid)}(), boundary_identifiers(g.grids[i]))
    end
    return LazyTensors.concatenate_tuples(per_grid...)
end


"""
    boundary_grid(g::TensorGrid, id::TensorGridBoundary)

The grid for the boundary of `g` specified by `id`.
"""
function boundary_grid(g::TensorGrid, id::TensorGridBoundary)
    local_boundary_grid = boundary_grid(g.grids[grid_id(id)], boundary_id(id))
    new_grids = Base.setindex(g.grids, local_boundary_grid, grid_id(id))
    return TensorGrid(new_grids...)
end


function combined_coordinate_vector_type(coordinate_types...)
    combined_coord_length = mapreduce(_ncomponents, +, coordinate_types)
    combined_coord_type = mapreduce(eltype, promote_type, coordinate_types)

    if combined_coord_length == 1
        return combined_coord_type
    else
        return SVector{combined_coord_length, combined_coord_type}
    end
end

function combine_coordinates(coords...)
    return mapreduce(SVector, vcat, coords)
end
