"""
    BoundaryCondition

Description of a boundary condition. Implementations describe the kind of
boundary condition, what boundary the condition applies to, and any associated
data. Should implement [`boundary`](@ref) and may implement
[`boundary_data`](@ref) if applicable.

For examples see [`DirichletCondition`](@ref) and [`NeumannCondition`](@ref)
"""
abstract type BoundaryCondition end

"""
    boundary(::BoundaryCondition)

The boundary identifier of the BoundaryCondition.
"""
function boundary end

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
struct DirichletCondition{DT,BID} <: BoundaryCondition
    data::DT
    boundary::BID
end
boundary_data(bc::DirichletCondition) = bc.data
boundary(bc::DirichletCondition) = bc.boundary

"""
    NeumannCondition{DT,BID}

A Neumann condition with `data::DT` on the boundary
specified by the boundary identifier `BID`.
"""
struct NeumannCondition{DT,BID} <: BoundaryCondition
    data::DT
    boundary::BID
end
boundary_data(bc::NeumannCondition) = bc.data
boundary(bc::NeumannCondition) = bc.boundary

