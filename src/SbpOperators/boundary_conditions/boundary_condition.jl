"""
    BoundaryCondition{BID}

A type for implementing data needed in order to impose a boundary condition.
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
    DirichletCondition{DT,BID}

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

