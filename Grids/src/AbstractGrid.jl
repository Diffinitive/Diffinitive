abstract type AbstractGrid end

function dimension(grid::AbstractGrid)
    error("Not implemented for abstact type AbstractGrid")
end

function points(grid::AbstractGrid)
    error("Not implemented for abstact type AbstractGrid")
end

# Evaluate function f on the grid g
function evalOn(g::AbstractGrid, f::Function)
    F(x) = f(x...)
    return F.(points(g))
end
