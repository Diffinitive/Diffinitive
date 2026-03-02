"""
    first_derivative(g, ..., [dim])

The first derivative operator `D1` as a `LazyTensor` on the given grid.

`D1` approximates the first-derivative d/dξ on `g` along the coordinate
dimension specified by `dim`.
"""
function first_derivative end

"""
    first_derivative(g::TensorGrid, stencil_set, dim)

See also: [`VolumeOperator`](@ref), [`LazyTensors.inflate`](@ref).
"""
function first_derivative(g::TensorGrid, stencil_set, dim)
    if dim ∉ 1:ndims(g)
        throw(DomainError(dim, "Direction must be inside [0, $(ndims(g))]."))
    end
    D₁ = first_derivative(g.grids[dim], stencil_set)
    return LazyTensors.inflate(D₁, size(g), dim)
end

function first_derivative(g::EquidistantGrid, stencil_set, dim)
    return first_derivative(TensorGrid(g), stencil_set, dim)
end

"""
    first_derivative(g::EquidistantGrid, stencil_set::StencilSet)

The first derivative operator on an `EquidistantGrid`. 
Uses the `D1` stencil in `stencil_set`.
"""
function first_derivative(g::EquidistantGrid, stencil_set::StencilSet)
    inner_stencil = parse_stencil(stencil_set["D1"]["inner_stencil"])
    closure_stencils = parse_stencil.(stencil_set["D1"]["closure_stencils"])
    return first_derivative(g, inner_stencil, closure_stencils);
end

"""
    first_derivative(g::EquidistantGrid, inner_stencil::Stencil, closure_stencils)

The first derivative operator on an `EquidistantGrid` given an
`inner_stencil` and `closure_stencils`.
"""
function first_derivative(g::EquidistantGrid, inner_stencil::Stencil, closure_stencils)
    h⁻¹ = inverse_spacing(g)
    return VolumeOperator(g, scale(inner_stencil,h⁻¹), scale.(closure_stencils,h⁻¹), odd)
end
