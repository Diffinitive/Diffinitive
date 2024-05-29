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
sat_tensors(Δ::Laplace, g::Grid, bc::DirichletCondition, tuning)

The operators required to construct the SAT for imposing a Dirichlet condition.
`tuning` specifies the strength of the penalty. See

See also: [`sat`,`DirichletCondition`, `positivity_decomposition`](@ref).
"""
function BoundaryConditions.sat_tensors(Δ::Laplace, g::Grid, bc::DirichletCondition, tuning)
    id = bc.id
    set  = Δ.stencil_set
    H⁻¹ = inverse_inner_product(g,set)
    Hᵧ = inner_product(boundary_grid(g, id), set)
    e = boundary_restriction(g, set, id)
    d = normal_derivative(g, set, id)
    B = positivity_decomposition(Δ, g, bc, tuning)
    sat_op = H⁻¹∘(d' - B*e')∘Hᵧ
    return sat_op, e
end
BoundaryConditions.sat_tensors(Δ::Laplace, g::Grid, bc::DirichletCondition) = BoundaryConditions.sat_tensors(Δ, g, bc, (1.,1.))

"""
sat_tensors(Δ::Laplace, g::Grid, bc::NeumannCondition)

The operators required to construct the SAT for imposing a Neumann condition


See also: [`sat`,`NeumannCondition`](@ref).
"""
function BoundaryConditions.sat_tensors(Δ::Laplace, g::Grid, bc::NeumannCondition)
    id = bc.id
    set  = Δ.stencil_set
    H⁻¹ = inverse_inner_product(g,set)
    Hᵧ = inner_product(boundary_grid(g, id), set)
    e = boundary_restriction(g, set, id)
    d = normal_derivative(g, set, id)

    sat_op = -H⁻¹∘e'∘Hᵧ
    return sat_op, d
end

function positivity_decomposition(Δ::Laplace, g::Grid, bc::DirichletCondition, tuning)
    pos_prop = positivity_properties(Δ)
    h = spacing(orthogonal_grid(g, bc.id))
    θ_H = pos_prop.theta_H
    τ_H = tuning[1]*ndims(g)/(h*θ_H)
    θ_R = pos_prop.theta_R
    τ_R = tuning[2]/(h*θ_R)
    B = τ_H + τ_R
    return B
end

positivity_properties(Δ::Laplace) = parse_named_tuple(Δ.stencil_set["Positivity"]["D2"]) # REVIEW: Can this function extract theta_H from the inner product instead of storing it twice in the TOML?
