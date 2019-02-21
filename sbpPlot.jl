module sbpPlot
using PyPlot, PyCall

function plotgridfunction(grid::EquidistantGrid, gridfunction)
    if dimension(grid) == 1
        plot(pointsalongdim(grid,1), gridfunction, linewidth=2.0)
    elseif dimension(grid) == 2
        mx = grid.size[1]
        my = grid.size[2]
        X = repeat(pointsalongdim(grid,1),1,my)
        Y = permutedims(repeat(pointsalongdim(grid,2),1,mx))
        plot_surface(X,Y,reshape(gridfunction,mx,my));
    else
        error(string("Plot not implemented for dimension ", string(dimension(grid))))
    end
end
end
