function SecondDerivative(grid::EquidistantGrid{Dim}, inner_stencil, closure_stencils, direction) where Dim
    h_inv = inverse_spacing(grid)[direction]
    return volume_operator(grid, scale(inner_stencil,h_inv^2), scale.(closure_stencils,h_inv^2), size(grid), even, direction)
end
SecondDerivative(grid::EquidistantGrid{1}, inner_stencil, closure_stencils) = SecondDerivative(grid,inner_stencil,closure_stencils,1)
export SecondDerivative
