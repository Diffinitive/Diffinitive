"""
    sat_tensors(op, grid, bc::BoundaryCondition, params...)

The tensor and boundary operator used to construct a simultaneous-approximation-term
for imposing `bc` related to `op`.

For `sat_op, L  = sat_tensors(...)` then `SAT = sat_op*(L*u - g)`  where `g` 
is the boundary data.
"""
function sat_tensors end


"""
    sat(op, grid, bc::BoundaryCondition, params...)

Simultaneous-Approximation-Term for a general `bc` to `op`. 
Returns a function `SAT(u,g)` weakly imposing `bc` when added to `op*u`.

`op` must implement the function `sat_tensors`.
"""
function sat(op, grid, bc::BoundaryCondition, params...)
    sat_op, L = sat_tensors(op, grid, bc, params...)
    return SAT(u, g) = sat_op*(L*u - g)
end