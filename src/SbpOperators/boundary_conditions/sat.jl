"""
    sat_tensors(op, grid, bc::BoundaryCondition; kwargs...)

The penalty tensor and boundary operator used to construct a
simultaneous-approximation-term for imposing `bc` related to `op`.

For `penalty_tensor, L  = sat_tensors(...)` then `SAT(u,g) =
penalty_tensor*(L*u - g)`  where `g` is the boundary data.
"""
function sat_tensors end


"""
    sat(op, grid, bc::BoundaryCondition; kwargs...)

Simultaneous-Approximation-Term for a general `bc` to `op`. Returns a function
`SAT(u,g)` weakly imposing `bc` when added to `op*u`.

Internally `sat_tensors(op, grid, bc; ...)` is called to construct the
necessary parts for the SAT.
"""
function sat(op, grid, bc::BoundaryCondition; kwargs...)
    penalty_tensor, L = sat_tensors(op, grid, bc; kwargs...)
    return SAT(u, g) = penalty_tensor*(L*u - g)
end
