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
    Laplace(g::Grid, stencil_set::StencilSet)

Creates the `Laplace` operator `Δ` on `g` given `stencil_set`. 

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

Creates the Laplace operator operator `Δ` as a `LazyTensor` on `g`.

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


"""
sat_tensors(Δ::Laplace, g::TensorGrid, bc::NeumannCondition)

Returns anonymous functions for construction the `LazyTensorApplication`s
recuired in order to impose a Neumann boundary condition.

See also: [`sat`,`NeumannCondition`](@ref).
"""
function BoundaryConditions.sat_tensors(Δ::Laplace, g::Grid, bc::NeumannCondition)
    id = bc.id
    set  = Δ.stencil_set
    H⁻¹ = inverse_inner_product(g,set)
    Hᵧ = inner_product(boundary_grid(g, id), set)
    e = boundary_restriction(g, set, id)
    d = normal_derivative(g, set, id)

    closure(u) = H⁻¹*e'*Hᵧ*d*u
    penalty(g) = -H⁻¹*e'*Hᵧ*g
    return closure, penalty
end
