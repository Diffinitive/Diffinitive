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

function laplace(grid::MappedGrid, stencil_set)
    J = jacobian_determinant(grid)
    J⁻¹ = DiagonalTensor(map(inv, J))

    Jg = map(*, J, geometric_tensor_inverse(grid))
    lg = logicalgrid(grid)

    return mapreduce(+, CartesianIndices(first(Jg))) do I
        i, j = I[1], I[2]
        Jgⁱʲ = componentview(Jg, i, j)

        if i == j
            J⁻¹∘second_derivative_variable(lg, Jgⁱʲ, stencil_set, i)
        else
            Dᵢ = first_derivative(lg, stencil_set, i)
            Dⱼ = first_derivative(lg, stencil_set, j)
            J⁻¹∘Dᵢ∘DiagonalTensor(Jgⁱʲ)∘Dⱼ
        end
    end
end


"""
    sat_tensors(Δ::Laplace, g::Grid, bc::DirichletCondition; H_tuning, R_tuning)

The operators required to construct the SAT for imposing a Dirichlet
condition. `H_tuning` and `R_tuning` are used to specify the strength of the
penalty.

See also: [`sat`](@ref),[`DirichletCondition`](@ref), [`positivity_decomposition`](@ref).
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

The operators required to construct the SAT for imposing a Neumann condition.

See also: [`sat`](@ref), [`NeumannCondition`](@ref).
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

"""
    positivity_decomposition(Δ::Laplace, g::Grid, bc::DirichletCondition; H_tuning, R_tuning)

Constructs the scalar `B` such that `d' - 1/2*B*e'` is symmetric positive
definite with respect to the boundary quadrature. Here `d` is the normal
derivative and `e` is the boundary restriction operator. `B` can then be used
to form a symmetric and energy stable penalty for a Dirichlet condition. The
parameters `H_tuning` and `R_tuning` are used to specify the strength of the
penalty and must be greater than 1. For details we refer to
https://doi.org/10.1016/j.jcp.2020.109294
"""
function positivity_decomposition(Δ::Laplace, g::Grid, bc::DirichletCondition; H_tuning, R_tuning)
    @assert(H_tuning ≥ 1.)
    @assert(R_tuning ≥ 1.)
    Nτ_H, τ_R = positivity_limits(Δ,g,bc)
    return H_tuning*Nτ_H + R_tuning*τ_R
end

# TODO: We should consider implementing a proper BoundaryIdentifier for EquidistantGrid and then
# change bc::BoundaryCondition to id::BoundaryIdentifier
function positivity_limits(Δ::Laplace, g::EquidistantGrid, bc::DirichletCondition)
    h = spacing(g)
    θ_H = parse_scalar(Δ.stencil_set["H"]["closure"][1])
    θ_R = parse_scalar(Δ.stencil_set["D2"]["positivity"]["theta_R"])

    τ_H = 1/(h*θ_H)
    τ_R = 1/(h*θ_R)
    return τ_H, τ_R
end

function positivity_limits(Δ::Laplace, g::TensorGrid, bc::DirichletCondition)
    τ_H, τ_R = positivity_limits(Δ, g.grids[grid_id(boundary(bc))], bc)
    return τ_H*ndims(g), τ_R
end
