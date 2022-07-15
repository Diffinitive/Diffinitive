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
    dims(grid::Grid)

A range containing the dimensions of `grid`
"""
dims(grid::Grid) = 1:dim(grid)

"""
    evalOn(grid::Grid, f::Function)

Evaluate function `f` on `grid`
"""
function evalOn(grid::Grid, f::Function)
    F(x) = f(x...)
    return F.(points(grid))
end