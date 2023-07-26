"""
    sat_tensors(op, grid, bc::BoundaryCondition, params...)

Returns the functions `closure(u)` and `penalty(g)` used to construct a SAT for the
`LazyTensor` operator `op` on `grid` associated with the boundary condition `bc`,
where g is the discretized data of `bc`.
"""
function sat_tensors end


"""
    sat(op, grid, bc::BoundaryCondition, params...)

Simultaneous-Approximation-Term for a general `BoundaryCondition` `bc` to `LazyTensor` `op`. 
The function returns a function `f`, where f(t,u)` returns a `LazyTensorApplication`
weakly imposing the boundary condition at time `t`, when added to `op*u`.

`op` must implement the function `sat_tensors`. `f` is then constructed as 
`f(t,u) = closure(u) + `penalty(g(t))`.
"""
function sat(op, grid, bc::BoundaryCondition, params...)
    sat_op, L = sat_tensors(op, grid, bc, params...)
    return SAT(u, g) = sat_op*(L*u - g)
end