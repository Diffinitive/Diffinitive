"""
    BoundaryCondition{BID}

Description of a boundary condition. Implementations describe the kind of
boundary condition, what boundary the condition applies to, and any associated
data. Should implement [`boundary`](@ref) and may implement
[`boundary_data`](@ref) if applicable.

For examples see [`DirichletCondition`](@ref) and [`NeumannCondition`](@ref)
"""
abstract type BoundaryCondition{BID} end

"""
    boundary(::BoundaryCondition)

The boundary identifier of the BoundaryCondition.
"""
boundary(::BoundaryCondition{BID}) where {BID} = BID()

"""
    boundary_data(::BoundaryCondition)

If implemented, the data associated with the BoundaryCondition.
"""
function boundary_data end

"""
    discretize_data(grid, bc::BoundaryCondition)

The data of `bc` as a lazily evaluated grid function on the boundary grid
specified by `boundary(bc)`.
"""
function discretize_data(grid, bc::BoundaryCondition)
    return eval_on(boundary_grid(grid, boundary(bc)), boundary_data(bc))
end

"""
    DirichletCondition{DT,BID}

A Dirichlet condition with `data::DT` on the boundary
specified by the boundary identifier `BID`.
"""
struct DirichletCondition{DT,BID} <: BoundaryCondition{BID}
    data::DT
    function DirichletCondition(data, id)
        return new{typeof(data),typeof(id)}(data)
    end
end
boundary_data(bc::DirichletCondition) = bc.data

"""
    NeumannCondition{DT,BID}

A Neumann condition with `data::DT` on the boundary
specified by the boundary identifier `BID`.
"""
struct NeumannCondition{DT,BID} <: BoundaryCondition{BID}
    data::DT
    function NeumannCondition(data, id)
        return new{typeof(data),typeof(id)}(data)
    end
end
boundary_data(bc::NeumannCondition) = bc.data

