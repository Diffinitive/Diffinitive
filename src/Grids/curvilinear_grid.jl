# TBD: Rename to MappedGrid?
struct CurvilinearGrid{T,D, GT<:Grid{<:Any,D}, CT<:AbstractArray{T,D}, JT<:AbstractArray{<:AbstractArray{<:Any, 2}, D}} <: Grid{T,D}
    logicalgrid::GT
    physicalcoordinates::CT
    jacobian::JT # TBD: currectly ∂xᵢ/∂ξⱼ. Is this the correct index order?
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
boundary_indices(g::CurvilinearGrid, id::TensorGridBoundary) = boundary_indices(g.logicalgrid, id)

function boundary_grid(g::CurvilinearGrid, id::TensorGridBoundary)
    b_indices = boundary_indices(g.logicalgrid, id)

    # Calculate indices of needed jacobian combonents
    D = ndims(g)
    all_indices = SVector{D}(1:D)
    free_variable_indices = deleteat(all_indices, grid_id(id))
    jacobian_components = (:, free_variable_indices)

    # Create grid function for boundary grid jacobian
    boundary_jacobian = componentview((@view g.jacobian[b_indices...])  , jacobian_components...)
    boundary_physicalcoordinates = @view g.physicalcoordinates[b_indices...]

    return CurvilinearGrid(
        boundary_grid(g.logicalgrid, id),
        boundary_physicalcoordinates,
        boundary_jacobian,
    )
end

# Do we add a convenience function `curvilinear_grid`? It could help with
# creating the logical grid, evaluating functions and possibly calculating the
# entries in the jacobian.

function curvilinear_grid(x, J, size...)
    D = length(size)
    lg = equidistant_grid(size, ntuple(i->0., D), ntuple(i->1., D))
    return CurvilinearGrid(
        lg,
        map(x,lg),
        map(J,lg),
    )
end
