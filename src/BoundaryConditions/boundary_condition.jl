"""
    BoundaryCondition{BID}

A type for implementing boundary_data needed in order to impose a boundary condition.
Subtypes refer to perticular types of boundary conditions, e.g. Neumann conditions.
"""
abstract type BoundaryCondition{BID} end

"""
    boundary(::BoundaryCondition)

The boundary identifier of the BoundaryCondition.
"""
boundary(::BoundaryCondition{BID}) where {BID} = BID()

"""
    boundary_data(::BoundaryCondition)

If implemented, the data associated with the BoundaryCondition
"""
function boundary_data end

"""
    discretize(grid, bc::BoundaryCondition)

The grid function obtained from discretizing the `bc` boundary_data on the boundary grid
    specified the by bc `id`.
"""
function discretize_data(grid, bc::BoundaryCondition)
    return eval_on(boundary_grid(grid, boundary(bc)), boundary_data(bc))
end

struct DirichletCondition{T1,T2} <: BoundaryCondition{T2}
    data::T1
    function DirichletCondition(data, id)
        return new{typeof(data),typeof(id)}(data)
    end
end
boundary_data(bc::DirichletCondition) = bc.data

struct NeumannCondition{T1,T2} <: BoundaryCondition{T2}
    data::T1
    function NeumannCondition(data, id)
        return new{typeof(data),typeof(id)}(data)
    end
end
boundary_data(bc::NeumannCondition) = bc.data

