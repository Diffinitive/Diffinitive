"""
    sat_tensors(op, grid, bc::BoundaryCondition{T}, params...)

Returns the `LazyTensor`s used to construct a SAT for the SBP operator `op` on 
`grid` associated with the boundary condition `bc`.
"""
function sat_tensors end

# TODO: Docs must be more specific in what this function does...
"""
    sat(op, grid, bc::BoundaryCondition, params...)

Simultaneous-Approximation-Term for general BoundaryCondition bc. f = sat(op, grid, bc) returns
an anonymous function, such that f(t,u) is a `LazyTensorApplication` weakly imposing bc
at time t.
"""
function sat(op, grid, bc::BoundaryCondition, params...)
    closure, penalty = sat_tensors(op, grid, bc, params...)
    data_array = discretize(data(bc),boundary_grid(grid, bc.id))
    return (t,u) -> closure(u) + penalty(data_array(t))
end

function sat(op, grid, bc::BoundaryCondition{ZeroBoundaryData}, params...)
    closure = sat_tensors(op, grid, bc, params...)
    return (t,u) -> closure(u)
end


# """
#     sat(op, grid, bc::BoundaryCondition{SpaceDependentBoundaryData{T}}, params...)

# Simultaneous-Approximation-Term for space-dependent boundary data. f = sat(op, grid, bc) returns
# an anonymous function, such that f(u) is a `LazyTensorApplication` weakly imposing the BC.
# """
# function sat(op, grid, bc::BoundaryCondition{SpaceDependentBoundaryData{T}}, params...) where T
#     closure, penalty = sat_tensors(op, grid, bc, params...)
#     g = data(bc)
#     return u -> closure(u) + penalty(g)
# end

# """
#     sat(op, grid, bc::BoundaryCondition{SpaceDependentBoundaryData{T}}, params...)

# Simultaneous-Approximation-Term for time-dependent boundary data. f = sat(op, grid, bc) returns
# an anonymous function, such that f(u,t) is a `LazyTensorApplication` weakly imposing the BC at time t.
# """
# function sat(op, grid, bc::BoundaryCondition{TimeDependentBoundaryData{T}}, params...) where T
#     closure, penalty = sat_tensors(op, grid, bc, params...)
#     b_sz = size(boundary_grid(grid, bc.id))
#     b_vec = ones(eltype(grid), b_sz)
#     g = data(bc)
#     return (u,t) -> closure(u) + g(t)*penalty(b_vec)
# end

# """
#     sat(op, grid, bc::BoundaryCondition{SpaceDependentBoundaryData{T}}, params...)

# Simultaneous-Approximation-Term for space-time-dependent boundary data. f = sat(op, grid, bc) returns
# an anonymous function, such that f(u,t) is a `LazyTensorApplication` weakly imposing the BC at time t.
# """
# function sat(op, grid, bc::BoundaryCondition{SpaceTimeDependentBoundaryData{T}}, params...) where T
#     closure, penalty = sat_tensors(op, grid, bc, params...)
#     g = data(bc)
#     return (u,t) -> closure(u) + penalty(g(t))
# end