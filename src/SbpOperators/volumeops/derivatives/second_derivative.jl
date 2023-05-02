"""
    second_derivative(g::EquidistantGrid, stencil_set, direction)

Creates the second-derivative operator `D2` as a `LazyTensor`

`D2` approximates the second-derivative d²/dξ² on `g` along the coordinate
dimension specified by `direction`.

See also: [`VolumeOperator`](@ref), [`LazyTensors.inflate`](@ref).
"""
function second_derivative(g::TensorGrid, stencil_set, direction)
    D₂ = second_derivative(g.grids[direction], stencil_set)
    return LazyTensors.inflate(D₂, size(g), direction)
end

"""
    second_derivative(g, stencil_set)

Creates a `second_derivative` operator on a 1D `g` given a `stencil_set`. Uses
the `D2` stencil in the stencil set.
"""
function second_derivative(g::EquidistantGrid, stencil_set::StencilSet)
    inner_stencil = parse_stencil(stencil_set["D2"]["inner_stencil"])
    closure_stencils = parse_stencil.(stencil_set["D2"]["closure_stencils"])
    return second_derivative(g, inner_stencil, closure_stencils)
end

"""
    second_derivative(g, inner_stencil, closure_stencils)

Creates a `second_derivative` operator on a 1D `g` given `inner_stencil` and
`closure_stencils`.
"""
function second_derivative(g::EquidistantGrid, inner_stencil::Stencil, closure_stencils)
    h⁻¹ = inverse_spacing(g)
    return VolumeOperator(g, scale(inner_stencil,h⁻¹^2), scale.(closure_stencils,h⁻¹^2), even)
end
