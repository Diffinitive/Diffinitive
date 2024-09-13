"""
    MappedGrid{T,D} <: Grid{T,D}

A grid defined by a coordinate mapping from a logical grid to some physical
coordinates. The physical coordinates and the Jacobian are stored as grid
functions corresponding to the logical grid.

See also: [`logicalgrid`](@ref), [`jacobian`](@ref), [`metric_tensor`](@ref).
"""
struct MappedGrid{T,D, GT<:Grid{<:Any,D}, CT<:AbstractArray{T,D}, JT<:AbstractArray{<:AbstractArray{<:Any, 2}, D}} <: Grid{T,D}
    logicalgrid::GT
    physicalcoordinates::CT
    jacobian::JT

    """
        MappedGrid(logicalgrid, physicalcoordinates, jacobian)

    A MappedGrid with the given physical coordinates and jacobian.
    """
    function MappedGrid(logicalgrid::GT, physicalcoordinates::CT, jacobian::JT) where {T,D, GT<:Grid{<:Any,D}, CT<:AbstractArray{T,D}, JT<:AbstractArray{<:AbstractArray{<:Any, 2}, D}}
        if !(size(logicalgrid) == size(physicalcoordinates) == size(jacobian))
            throw(ArgumentError("Sizes must match"))
        end

        if size(first(jacobian)) != (length(first(physicalcoordinates)),D)
            throw(ArgumentError("The size of the jacobian must match the dimensions of the grid and coordinates"))
        end

        return new{T,D,GT,CT,JT}(logicalgrid, physicalcoordinates, jacobian)
    end
end

function Base.:(==)(a::MappedGrid, b::MappedGrid)
    same_logicalgrid = logicalgrid(a) == logicalgrid(b)
    same_coordinates = collect(a) == collect(b)
    same_jacobian = jacobian(a) == jacobian(b)

    return same_logicalgrid && same_coordinates && same_jacobian
end

# Review: rename function logicalgrid to logical_grid
# for consistency with mapped_grid. 
"""
    logicalgrid(g::MappedGrid)

The logical grid of `g`.
"""
logicalgrid(g::MappedGrid) = g.logicalgrid

"""
    jacobian(g::MappedGrid)

The Jacobian matrix of `g` as a grid function.
"""
jacobian(g::MappedGrid) = g.jacobian


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


"""
    mapped_grid(x, J, size::Vararg{Int})

A `MappedGrid` with a default logical grid on a unit hyper box. `x` and `J`
are functions to be evaluated on the logical grid and `size` determines the
size of the logical grid.
"""
function mapped_grid(x, J, size::Vararg{Int})
    D = length(size)
    lg = equidistant_grid(ntuple(i->0., D), ntuple(i->1., D), size...)
    return mapped_grid(lg, x, J)
end

"""
    mapped_grid(lg::Grid, x, J)

A `MappedGrid` with logical grid `lg`. Physical coordinates and Jacobian are
determined by the functions `x` and `J`.
"""
function mapped_grid(lg::Grid, x, J)
    return MappedGrid(
        lg,
        map(x,lg),
        map(J,lg),
    )
end

"""
    jacobian_determinant(g::MappedGrid)

The jacobian determinant of `g` as a grid function.
"""
function jacobian_determinant(g::MappedGrid)
    return map(jacobian(g)) do ∂x∂ξ
        det(∂x∂ξ)
    end
end
# TBD: Should this be changed to calculate sqrt(g) instead?
#       This would make it well defined also for n-dim grids embedded in higher dimensions.
# TBD: Is there a better name? metric_determinant?
# TBD: Is the best option to delete it?
# Review: I don't think we should delete it. Users building their own 
#         curvilinear operators will need the functionality. Also the
#         determinant of the jacobian (and not its square root) is required
#         for quadratures on mapped grids right? For that reason I think we should
#         keep the function as is. We could provide a function for the square root
#         as well if we think it would be helpfull. Regarding naming, perhaps
#         metric_determinant is better?

"""
    metric_tensor(g::MappedGrid)

The metric tensor of `g` as a grid function.
"""
function metric_tensor(g::MappedGrid)
    return map(jacobian(g)) do ∂x∂ξ
        ∂x∂ξ'*∂x∂ξ
    end
end

"""
    metric_tensor_inverse(g::MappedGrid)

The inverse of the metric tensor of `g` as a grid function.
"""
function metric_tensor_inverse(g::MappedGrid)
    return map(jacobian(g)) do ∂x∂ξ
        inv(∂x∂ξ'*∂x∂ξ)
    end
end

function min_spacing(g::MappedGrid{T,1} where T)
    n, = size(g)

    ms = Inf
    for i ∈ 1:n-1
        ms = min(ms, norm(g[i+1]-g[i]))
    end

    return ms
end

function min_spacing(g::MappedGrid{T,2} where T)
    n, m = size(g)

    ms = Inf
    for i ∈ 1:n-1, j ∈ 1:m-1 # loop over each cell of the grid

        ms = min(
            ms,
            norm(g[i+1,j]-g[i,j]),
            norm(g[i,j+1]-g[i,j]),

            norm(g[i+1,j]-g[i+1,j+1]),
            norm(g[i,j+1]-g[i+1,j+1]),

            norm(g[i+1,j+1]-g[i,j]),
            norm(g[i+1,j]-g[i,j+1]),
        )
        # NOTE: This could be optimized to avoid checking all interior edges twice.
    end

    return ms
end

"""
    normal(g::MappedGrid, boundary)

The outward pointing normal as a grid function on the corresponding boundary grid.
"""
function normal(g::MappedGrid, boundary)
    b_indices = boundary_indices(g, boundary)
    σ =_boundary_sign(component_type(g), boundary)
    return map(jacobian(g)[b_indices...]) do ∂x∂ξ
        ∂ξ∂x = inv(∂x∂ξ)
        k = grid_id(boundary)
        σ*∂ξ∂x[k,:]/norm(∂ξ∂x[k,:])
    end
end

function _boundary_sign(T, boundary)
    if boundary_id(boundary) == UpperBoundary()
        return one(T)
    elseif boundary_id(boundary) == LowerBoundary()
        return -one(T)
    else
        throw(ArgumentError("The boundary identifier must be either `LowerBoundary()` or `UpperBoundary()`"))
    end
end
