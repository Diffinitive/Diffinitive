"""
second_derivative(g::TensorGrid, stencil_set, dim)
second_derivative(g::EquidistantGrid, stencil_set, dim)

Creates the second derivative operator `D2` as a `LazyTensor`

`D2` approximates the second-derivative d²/dξ² on `g` along the coordinate
dimension specified by `dim`.

See also: [`VolumeOperator`](@ref), [`LazyTensors.inflate`](@ref).
"""
function second_derivative(g::TensorGrid, stencil_set, dim)
    if dim ∉ 1:ndims(g)
        throw(DomainError(dim, "Direction must be inside [0, $(ndims(g))]."))
    end
    D₂ = second_derivative(g.grids[dim], stencil_set)
    return LazyTensors.inflate(D₂, size(g), dim)
end

function second_derivative(g::EquidistantGrid, stencil_set::StencilSet, dim)
    return second_derivative(TensorGrid(g), stencil_set, dim)
end

"""
    second_derivative(g::EquidistantGrid, stencil_set::::StencilSet)

The second derivative operator on an `EquidistantGrid`. 
Uses the `D2` stencil in `stencil_set`.
"""
function second_derivative(g::EquidistantGrid, stencil_set::StencilSet)
    inner_stencil = parse_stencil(stencil_set["D2"]["inner_stencil"])
    closure_stencils = parse_stencil.(stencil_set["D2"]["closure_stencils"])
    return second_derivative(g, inner_stencil, closure_stencils)
end

"""
    second_derivative(g::EquidistantGrid, inner_stencil::Stencil, closure_stencils)

The second derivative operator on an `EquidistantGrid`, given `inner_stencil` and
`closure_stencils`.
"""
function second_derivative(g::EquidistantGrid, inner_stencil::Stencil, closure_stencils)
    h⁻¹ = inverse_spacing(g)
    return VolumeOperator(g, scale(inner_stencil,h⁻¹^2), scale.(closure_stencils,h⁻¹^2), even)
end
