"""
    Laplace{T, Dim, TM} <: LazyTensor{T, Dim, Dim}

The Laplace operator, approximating ∑d²/xᵢ² , i = 1,...,`Dim` as a
`LazyTensor`.
"""
struct Laplace{T, Dim, TM<:LazyTensor{T, Dim, Dim}} <: LazyTensor{T, Dim, Dim}
    D::TM       # Difference operator
    stencil_set::StencilSet # Stencil set of the operator
end

"""
    Laplace(grid::Equidistant, stencil_set)

Creates the `Laplace` operator `Δ` on `grid` given a `stencil_set`. 

See also [`laplace`](@ref).
"""
function Laplace(g::Grid, stencil_set::StencilSet)
    Δ = laplace(g, stencil_set)
    return Laplace(Δ, stencil_set)
end

LazyTensors.range_size(L::Laplace) = LazyTensors.range_size(L.D)
LazyTensors.domain_size(L::Laplace) = LazyTensors.domain_size(L.D)
LazyTensors.apply(L::Laplace, v::AbstractArray, I...) = LazyTensors.apply(L.D,v,I...)

# TODO: Implement pretty printing of Laplace once pretty printing of LazyTensors is implemented.
# Base.show(io::IO, L::Laplace) = ...

"""
    laplace(g::Grid, stencil_set)

Creates the Laplace operator operator `Δ` as a `LazyTensor` on the given grid.

`Δ` approximates the Laplace operator ∑d²/xᵢ² , i = 1,...,`Dim` on `g`. The
approximation depends on the type of grid and the stencil set.

See also: [`second_derivative`](@ref).
"""
function laplace end
function laplace(g::TensorGrid, stencil_set)
    # return mapreduce(+, enumerate(g.grids)) do (i, gᵢ)
    #     Δᵢ = laplace(gᵢ, stencil_set)
    #     LazyTensors.inflate(Δᵢ, size(g), i)
    # end

    Δ = LazyTensors.inflate(laplace(g.grids[1], stencil_set), size(g), 1)
    for d = 2:ndims(g)
        Δ += LazyTensors.inflate(laplace(g.grids[d], stencil_set), size(g), d)
    end
    return Δ
end
laplace(g::EquidistantGrid, stencil_set) = second_derivative(g, stencil_set)
