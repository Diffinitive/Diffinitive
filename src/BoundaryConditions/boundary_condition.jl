"""
    BoundaryDataType

A type for storing boundary data, e.g. constant, space-dependent, time-dependent etc.
Subtypes of `BoundaryDataType` should store the boundary data in a field `val`, i.e.
`struct MyBoundaryDataType{T} <: BoundaryDataType val::T end`.
"""
abstract type BoundaryDataType end

struct ConstantBoundaryData{T} <: BoundaryDataType
    val::T
end

struct SpaceDependentBoundaryData{T} <: BoundaryDataType
    val::T
end

struct TimeDependentBoundaryData{T} <: BoundaryDataType
    val::T
end

struct SpaceTimeDependentBoundaryData{T} <: BoundaryDataType
    val::T
end

"""
    BoundaryCondition

A type for implementing data needed in order to impose a boundary condition.
Subtypes refer to perticular types of boundary conditions, e.g. Neumann conditions.
"""
# TODO: Parametrize the boundary id as well?
abstract type BoundaryCondition{T<:BoundaryDataType} end

data(bc::BoundaryCondition) = bc.data.val

struct NeumannCondition{BDT<:BoundaryDataType} <: BoundaryCondition{BDT}
    id::BoundaryIdentifier
    data::BDT
end

struct DirichletCondition{BDT<:BoundaryDataType} <: BoundaryCondition{BDT}
    id::BoundaryIdentifier
    data::BDT
end

struct RobinCondition{BDT<:BoundaryDataType,T<:Real} <: BoundaryCondition{BDT}
    id::BoundaryIdentifier
    data::BDT
    α::T
    β::T
end