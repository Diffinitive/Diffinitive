"""
     Grid

Should implement
    dim(grid::Grid)
    points(grid::Grid)

"""
abstract type Grid end
function dim end # TODO: Rename to Base.ndims instead? That's the name used for arrays.
function points end

"""
    dims(g::Grid)

A range containing the dimensions of the grid
"""
dims(grid::Grid) = 1:dim(grid)

"""
    evalOn(g::Grid, f::Function)

Evaluate function f on the grid g
"""
function evalOn(g::Grid, f::Function)
    F(x) = f(x...)
    return F.(points(g))
end