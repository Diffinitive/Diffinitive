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
sat_tensors(Δ::Laplace, g::Grid, bc::DirichletCondition; tuning)

The operators required to construct the SAT for imposing a Dirichlet condition.
`tuning` specifies the strength of the penalty. See

See also: [`sat`,`DirichletCondition`, `positivity_decomposition`](@ref).
"""
function sat_tensors(Δ::Laplace, g::Grid, bc::DirichletCondition; H_tuning = 1., R_tuning = 1.)
    id = boundary(bc)
    set  = Δ.stencil_set
    H⁻¹ = inverse_inner_product(g,set)
    Hᵧ = inner_product(boundary_grid(g, id), set)
    e = boundary_restriction(g, set, id)
    d = normal_derivative(g, set, id)
    B = positivity_decomposition(Δ, g, bc; H_tuning, R_tuning)
    penalty_tensor = H⁻¹∘(d' - B*e')∘Hᵧ
    return penalty_tensor, e
end

"""
sat_tensors(Δ::Laplace, g::Grid, bc::NeumannCondition)

The operators required to construct the SAT for imposing a Neumann condition


See also: [`sat`,`NeumannCondition`](@ref).
"""
function sat_tensors(Δ::Laplace, g::Grid, bc::NeumannCondition)
    id = boundary(bc)
    set  = Δ.stencil_set
    H⁻¹ = inverse_inner_product(g,set)
    Hᵧ = inner_product(boundary_grid(g, id), set)
    e = boundary_restriction(g, set, id)
    d = normal_derivative(g, set, id)

    penalty_tensor = -H⁻¹∘e'∘Hᵧ
    return penalty_tensor, d
end

# TODO: We should consider implementing a proper BoundaryIdentifier for EquidistantGrid and then
# change bc::BoundaryCondition to id::BoundaryIdentifier

function positivity_decomposition(Δ::Laplace, g::EquidistantGrid, bc::BoundaryCondition; H_tuning, R_tuning)
    pos_prop = positivity_properties(Δ)
    h = spacing(g)
    θ_H = pos_prop.theta_H
    τ_H = H_tuning*ndims(g)/(h*θ_H)
    θ_R = pos_prop.theta_R
    τ_R = R_tuning/(h*θ_R)
    B = τ_H + τ_R
    return B
end

function positivity_decomposition(Δ::Laplace, g::TensorGrid, bc::BoundaryCondition; H_tuning, R_tuning)
    pos_prop = positivity_properties(Δ)
    h = spacing(g.grids[grid_id(boundary(bc))]) # grid spacing of the 1D grid normal to the boundary
    θ_H = pos_prop.theta_H
    τ_H = H_tuning*ndims(g)/(h*θ_H)
    θ_R = pos_prop.theta_R
    τ_R = R_tuning/(h*θ_R)
    B = τ_H + τ_R
    return B
end

function positivity_properties(Δ::Laplace)
    D2_pos_prop = parse_named_tuple(Δ.stencil_set["D2"]["positivity"])
    H_closure = parse_tuple(Δ.stencil_set["H"]["closure"])
    return merge(D2_pos_prop, (theta_H = H_closure[1],))
end
