abstract type BoundaryConditionType end
struct Dirichlet <: BoundaryConditionType end
struct Neumann <: BoundaryConditionType end

struct BoundaryCondition{BCType <: BoundaryConditionType, ID <: BoundaryIdentifier, DType}
    id::ID
    data::DType
    BoundaryCondition{BCType}(id::ID, data::DType) where {BCType <: BoundaryConditionType,
                                                          ID <: BoundaryIdentifier,
                                                          DType} = new{BCType,ID,DType}(id, data)
end