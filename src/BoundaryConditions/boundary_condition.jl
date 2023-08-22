"""
    BoundaryCondition

A type for implementing data needed in order to impose a boundary condition.
Subtypes refer to perticular types of boundary conditions, e.g. Neumann conditions.
"""
abstract type BoundaryCondition{T} end

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
    discretize(::BoundaryData, boundary_grid)

Returns an anonymous time-dependent function f, such that f(t) is
a `LazyArray` holding the `BoundaryData` discretized on `boundary_grid`.
"""
function discretize_data(grid, bc::BoundaryCondition)
    return eval_on(boundary_grid(grid, id(bc)), data(bc))
end

struct DirichletCondition{T} <: BoundaryCondition{T}
    data::T
    id::BoundaryIdentifier
end
id(bc::DirichletCondition) = bc.id
data(bc::DirichletCondition) = bc.data

struct NeumannCondition{T} <: BoundaryCondition{T}
    data::T
    id::BoundaryIdentifier 
end
id(bc::NeumannCondition) = bc.id
data(bc::NeumannCondition) = bc.data

