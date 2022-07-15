"""
     AbstractGrid

Should implement
    dimension(grid::AbstractGrid)
    points(grid::AbstractGrid)

"""
abstract type AbstractGrid end
export AbstractGrid
function dimension end
function points end
export dimension, points

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
dims(grid::AbstractGrid) = 1:dimension(grid)
export dims