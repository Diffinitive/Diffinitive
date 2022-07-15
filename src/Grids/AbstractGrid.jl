"""
     AbstractGrid

Should implement
    dim(grid::AbstractGrid)
    points(grid::AbstractGrid)

"""
abstract type AbstractGrid end
export AbstractGrid
function dim end # TODO: Rename to Base.ndims instead? That's the name used for arrays.
function points end
export dim, points

"""
    evalOn(g::AbstractGrid, f::Function)

Evaluate function f on the grid g
"""
function evalOn(g::AbstractGrid, f::Function)
    F(x) = f(x...)
    return F.(points(g))
end
export evalOn


"""
    dims(g::AbstractGrid)

A range containing the dimensions of the grid
"""
dims(grid::AbstractGrid) = 1:dim(grid)
export dims