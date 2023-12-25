"""
    BoundaryCondition

A type for implementing data needed in order to impose a boundary condition.
Subtypes refer to perticular types of boundary conditions, e.g. Neumann conditions.
"""
abstract type BoundaryCondition{T1,T2} end

"""
    id(::BoundaryCondition)

The boundary identifier of the BoundaryCondition.
Must be implemented by subtypes.
"""
function id end

"""
    data(::BoundaryCondition)

If implemented, the data associated with the BoundaryCondition
"""
function data end

"""
    discretize(grid, bc::BoundaryCondition)

The grid function obtained from discretizing the `bc` data on the boundary grid
    specified the by bc `id`.
"""
function discretize_data(grid, bc::BoundaryCondition)
    return eval_on(boundary_grid(grid, id(bc)), data(bc))
end

struct DirichletCondition{T1,T2} <: BoundaryCondition{T1,T2}
    data::T1
    id::T2
end
id(bc::DirichletCondition) = bc.id
data(bc::DirichletCondition) = bc.data

struct NeumannCondition{T1,T2} <: BoundaryCondition{T1,T2}
    data::T1
    id::T2
end
id(bc::NeumannCondition) = bc.id
data(bc::NeumannCondition) = bc.data

