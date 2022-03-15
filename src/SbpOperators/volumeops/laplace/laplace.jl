"""
    Laplace{T, Dim, TM} <: TensorMapping{T, Dim, Dim}

Implements the Laplace operator, approximating ∑d²/xᵢ² , i = 1,...,`Dim` as a
`TensorMapping`. Additionally `Laplace` stores the stencil set (parsed from TOML) 
used to construct the `TensorMapping`.
"""
struct Laplace{T, Dim, TM<:TensorMapping{T, Dim, Dim}} <: TensorMapping{T, Dim, Dim}
    D::TM       # Difference operator
    stencil_set # Stencil set of the operator
end

"""
    Laplace(grid::Equidistant, stencil_set)

Creates the `Laplace` operator `Δ` on `grid` given a parsed TOML
`stencil_set`. See also [`laplace`](@ref).
"""
function Laplace(grid::EquidistantGrid, stencil_set)
    inner_stencil = parse_stencil(stencil_set["D2"]["inner_stencil"])
    closure_stencils = parse_stencil.(stencil_set["D2"]["closure_stencils"])
    Δ = laplace(grid, inner_stencil,closure_stencils)
    return Laplace(Δ,stencil_set)
end

LazyTensors.range_size(L::Laplace) = LazyTensors.range_size(L.D)
LazyTensors.domain_size(L::Laplace) = LazyTensors.domain_size(L.D)
LazyTensors.apply(L::Laplace, v::AbstractArray, I...) = LazyTensors.apply(L.D,v,I...)

# TODO: Implement pretty printing of Laplace once pretty printing of TensorMappings is implemented. 
# Base.show(io::IO, L::Laplace) = ...

"""
    laplace(grid::EquidistantGrid, inner_stencil, closure_stencils)

Creates the Laplace operator operator `Δ` as a `TensorMapping`

`Δ` approximates the Laplace operator ∑d²/xᵢ² , i = 1,...,`Dim` on `grid`, using
the stencil `inner_stencil` in the interior and a set of stencils `closure_stencils`
for the points in the closure regions.

On a one-dimensional `grid`, `Δ` is equivalent to `second_derivative`. On a
multi-dimensional `grid`, `Δ` is the sum of multi-dimensional `second_derivative`s
where the sum is carried out lazily.

See also: [`second_derivative`](@ref).
"""
function laplace(grid::EquidistantGrid, inner_stencil, closure_stencils)
    Δ = second_derivative(grid, inner_stencil, closure_stencils, 1)
    for d = 2:dimension(grid)
        Δ += second_derivative(grid, inner_stencil, closure_stencils, d)
    end
    return Δ
end
