"""
    first_derivative(g, ..., [direction])

The first-derivative operator `D1` as a `LazyTensor` on the given grid.

`D1` approximates the first-derivative d/dξ on `g` along the coordinate
dimension specified by `direction`.
"""
function first_derivative end

"""
    first_derivative(g::TensorGrid, stencil_set, direction)

See also: [`VolumeOperator`](@ref), [`LazyTensors.inflate`](@ref).
"""
function first_derivative(g::TensorGrid, stencil_set, direction)
    D₁ = first_derivative(g.grids[direction], stencil_set)
    return LazyTensors.inflate(D₁, size(g), direction)
end

"""
    first_derivative(g::EquidistantGrid, stencil_set)

The first derivative operator on a `EquidistantGrid` given a
`StencilSet`. Uses the `D1` stencil in the stencil set.
"""
function first_derivative(g::EquidistantGrid, stencil_set::StencilSet)
    inner_stencil = parse_stencil(stencil_set["D1"]["inner_stencil"])
    closure_stencils = parse_stencil.(stencil_set["D1"]["closure_stencils"])
    return first_derivative(g, inner_stencil, closure_stencils);
end

"""
    first_derivative(g::EquidistantGrid, inner_stencil, closure_stencils)

The first derivative operator on a `EquidistantGrid` given an
`inner_stencil` and a`closure_stencils`.
"""
function first_derivative(g::EquidistantGrid, inner_stencil::Stencil, closure_stencils)
    h⁻¹ = inverse_spacing(g)
    return VolumeOperator(g, scale(inner_stencil,h⁻¹), scale.(closure_stencils,h⁻¹), odd)
end
