include("sbp.jl")
using Makie
import .sbp.Grid
function plotgridfunction(grid::sbp.Grid.EquidistantGrid, gridfunction::AbstractArray)
    if sbp.Grid.dimension(grid) == 1
        plot(sbp.Grid.pointsalongdim(grid,1), gridfunction)
    elseif sbp.Grid.dimension(grid) == 2
        scene = surface(sbp.Grid.pointsalongdim(grid,1),sbp.Grid.pointsalongdim(grid,2), gridfunction)
    else
        error(string("Plot not implemented for dimension ", string(dimension(grid))))
    end
end
