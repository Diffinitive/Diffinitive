struct MappedGrid{T,D, GT<:Grid{<:Any,D}, CT<:AbstractArray{T,D}, JT<:AbstractArray{<:AbstractArray{<:Any, 2}, D}} <: Grid{T,D}
    logicalgrid::GT
    physicalcoordinates::CT
    jacobian::JT
end

jacobian(g::MappedGrid) = g.jacobian
logicalgrid(g::MappedGrid) = g.logicalgrid


# Indexing interface
Base.getindex(g::MappedGrid, I::Vararg{Int}) = g.physicalcoordinates[I...]
Base.eachindex(g::MappedGrid) = eachindex(g.logicalgrid)

Base.firstindex(g::MappedGrid, d) = firstindex(g.logicalgrid, d)
Base.lastindex(g::MappedGrid, d) = lastindex(g.logicalgrid, d)

# Iteration interface

Base.iterate(g::MappedGrid) = iterate(g.physicalcoordinates)
Base.iterate(g::MappedGrid, state) = iterate(g.physicalcoordinates, state)

Base.IteratorSize(::Type{<:MappedGrid{<:Any, D}}) where D = Base.HasShape{D}()
Base.length(g::MappedGrid) = length(g.logicalgrid)
Base.size(g::MappedGrid) = size(g.logicalgrid)
Base.size(g::MappedGrid, d) = size(g.logicalgrid, d)

boundary_identifiers(g::MappedGrid) = boundary_identifiers(g.logicalgrid)
boundary_indices(g::MappedGrid, id::TensorGridBoundary) = boundary_indices(g.logicalgrid, id)

function boundary_grid(g::MappedGrid, id::TensorGridBoundary)
    b_indices = boundary_indices(g.logicalgrid, id)

    # Calculate indices of needed jacobian components
    D = ndims(g)
    all_indices = SVector{D}(1:D)
    free_variable_indices = deleteat(all_indices, grid_id(id))
    jacobian_components = (:, free_variable_indices)

    # Create grid function for boundary grid jacobian
    boundary_jacobian = componentview((@view g.jacobian[b_indices...])  , jacobian_components...)
    boundary_physicalcoordinates = @view g.physicalcoordinates[b_indices...]

    return MappedGrid(
        boundary_grid(g.logicalgrid, id),
        boundary_physicalcoordinates,
        boundary_jacobian,
    )
end

# TBD: refine and coarsen could be implemented once we have a simple manifold implementation.
# Before we do, we should consider the overhead of including such a field in the mapped grid struct.

function mapped_grid(x, J, size...)
    D = length(size)
    lg = equidistant_grid(ntuple(i->0., D), ntuple(i->1., D), size...)
    return MappedGrid(
        lg,
        map(x,lg),
        map(J,lg),
    )
end

function mapped_grid(c::Chart, size...)
    lg = equidistant_grid(parameterspace(c), size...)
    return MappedGrid(
        lg,
        map(c,lg),
        map(ξ->jacobian(c, ξ), lg),
    )
end

function jacobian_determinant(g::MappedGrid)
    return map(jacobian(g)) do ∂x∂ξ
        det(∂x∂ξ)
    end
end

function geometric_tensor(g::MappedGrid)
    return map(jacobian(g)) do ∂x∂ξ
        ∂x∂ξ'*∂x∂ξ
    end
end

function geometric_tensor_inverse(g::MappedGrid)
    return map(jacobian(g)) do ∂x∂ξ
        inv(∂x∂ξ'*∂x∂ξ)
    end
end

