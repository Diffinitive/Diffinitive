# TBD: Rename to MappedGrid?
struct CurvilinearGrid{T,D, GT<:Grid{<:Any,D}, CT<:AbstractArray{T,D}, JT<:AbstractArray{<:AbstractArray{<:Any, 2}, D}} <: Grid{T,D}
    logicalgrid::GT
    physicalcoordinates::CT
    jacobian::JT
end

jacobian(g::CurvilinearGrid) = g.jacobian
logicalgrid(g::CurvilinearGrid) = g.logicalgrid


# Indexing interface
Base.getindex(g::CurvilinearGrid, I::Vararg{Int}) = g.physicalcoordinates[I...]
Base.eachindex(g::CurvilinearGrid) = eachindex(g.logicalgrid)

Base.firstindex(g::CurvilinearGrid, d) = firstindex(g.logicalgrid, d)
Base.lastindex(g::CurvilinearGrid, d) = lastindex(g.logicalgrid, d)

# Iteration interface

Base.iterate(g::CurvilinearGrid) = iterate(g.physicalcoordinates)
Base.iterate(g::CurvilinearGrid, state) = iterate(g.physicalcoordinates, state)

Base.IteratorSize(::Type{<:CurvilinearGrid{<:Any, D}}) where D = Base.HasShape{D}()
Base.length(g::CurvilinearGrid) = length(g.logicalgrid)
Base.size(g::CurvilinearGrid) = size(g.logicalgrid)
Base.size(g::CurvilinearGrid, d) = size(g.logicalgrid, d)


# refine(g::TensorGrid, r::Int) = mapreduce(g->refine(g,r), TensorGrid, g.grids)
# coarsen(g::TensorGrid, r::Int) = mapreduce(g->coarsen(g,r), TensorGrid, g.grids)

# """
#     TensorGridBoundary{N, BID} <: BoundaryIdentifier

# A boundary identifier for a tensor grid. `N` Specifies which grid in the
# tensor product and `BID` which boundary on that grid.
# """
# struct TensorGridBoundary{N, BID} <: BoundaryIdentifier end
# grid_id(::TensorGridBoundary{N, BID}) where {N, BID} = N
# boundary_id(::TensorGridBoundary{N, BID}) where {N, BID} = BID()

# """
#     boundary_identifiers(g::TensorGrid)

# Returns a tuple containing the boundary identifiers of `g`.
# """
# function boundary_identifiers(g::TensorGrid)
#     per_grid = map(eachindex(g.grids)) do i
#         return map(bid -> TensorGridBoundary{i, typeof(bid)}(), boundary_identifiers(g.grids[i]))
#     end
#     return LazyTensors.concatenate_tuples(per_grid...)
# end


# """
#     boundary_grid(g::TensorGrid, id::TensorGridBoundary)

# The grid for the boundary of `g` specified by `id`.
# """
# function boundary_grid(g::TensorGrid, id::TensorGridBoundary)
#     local_boundary_grid = boundary_grid(g.grids[grid_id(id)], boundary_id(id))
#     new_grids = Base.setindex(g.grids, local_boundary_grid, grid_id(id))
#     return TensorGrid(new_grids...)
# end








# Do we add a convenience function `curvilinear_grid`? It could help with
# creating the logical grid, evaluating functions and possibly calculating the
# entries in the jacobian.

