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

boundary_identifiers(g::CurvilinearGrid) = boundary_identifiers(g.logicalgrid)

function boundary_grid(g::CurvilinearGrid, id::TensorGridBoundary)
    b_indices = boundary_indices(g.logicalgrid, id)
    return CurvilinearGrid(
        boundary_grid(g.logicalgrid, id),
        g.physicalcoordinates[b_indices],
        g.jacobian[b_indices],
    )
end



# Do we add a convenience function `curvilinear_grid`? It could help with
# creating the logical grid, evaluating functions and possibly calculating the
# entries in the jacobian.

