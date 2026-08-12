"""
    Elastic{Dim, TM} <: LazyTensor{Dim, Dim}

The isotropic elastic (Navier-Cauchy) differential operator approximating
∂ᵢλ∂ⱼ + ∂ⱼμ∂ᵢ + ∂ₖμ∂ₖδᵢⱼ, i, j = 1,..,`Dim`. as a `LazyTensor`.

"""
struct Elastic{Dim, TM<:LazyTensor{Dim, Dim}} <: LazyTensor{Dim, Dim}
    D::TM       # Difference operator
    stencil_set::StencilSet # Stencil set of the operator
end

"""
    Elastic(g::Grid, stencil_set::StencilSet)

Creates the `Elastic` operator with the first and second Lamé parameters
`λ` and `μ` on `g` given `stencil_set`.

See also [`elastic`](@ref).
"""
function Elastic(g::Grid, λ, μ, stencil_set::StencilSet)
    E = elastic(g, λ, μ, stencil_set)
    return Elastic(E, stencil_set)
end

LazyTensors.range_size(E::Elastic) = LazyTensors.range_size(E.D)
LazyTensors.domain_size(E::Elastic) = LazyTensors.domain_size(E.D)
LazyTensors.apply(E::Elastic, v::AbstractArray, I...) = LazyTensors.apply(E.D, v, I...)


"""
    elastic(g::Grid, λ, μ, stencil_set)

Creates the isotropic elastic (Navier-Cauchy) differential operator operator `E` with
first- and second Lamé parameters `λ` and `μ`, on `g` using operators from `stencil_set`.

`E` is the `MatrixTensor` approximating ∂ᵢλ∂ⱼ + ∂ⱼμ∂ᵢ + ∂ₖμ∂ₖδᵢⱼ, i,j = 1,...,`ndims(g)`.
For a displacement vector grid function ū `E*ū` approximates the divergence of the Cauchy
stress tensor, i.e., Eᵢⱼuⱼ = (∂ᵢλ∂ⱼ + ∂ⱼμ∂ᵢ + ∂ₖμ∂ₖδᵢⱼ) uⱼ

The approximation depends on the type of grid and the stencil set. It uses a combination of
narrow and wide second derivative approximations for improved dispersion properties.

See also: [`second_derivative_variable`](@ref), [`mixed_second_derivative_variable_wide`](@ref), 
[`mixed_second_derivative_variable_narrow`](@ref), [`MatrixTensor`](@ref)
"""
function elastic end
function elastic(g::TensorGrid, λ, μ, stencil_set)
    # Eᵢⱼuⱼ = ∂ᵢλ∂ⱼuⱼ + ∂ⱼμ∂ᵢuⱼ + ∂ₖμ∂ₖuᵢ
    #
    #       = ∂ᵢλ∂ⱼuⱼ + ∂ⱼμ∂ᵢuⱼ + ∂ₖμ∂ₖδᵢⱼuⱼ
    #
    #       =(∂ᵢλ∂ⱼ + ∂ⱼμ∂ᵢ + ∂ₖμ∂ₖδᵢⱼ) uⱼ

    N = ndims(g)

    Σₖ∂ₖμ∂ₖ = sum(1:N) do k
        second_derivative_variable(g,μ,stencil_set,k)
    end

    ∂∂_wide(i,σ,j) = mixed_second_derivative_variable_wide(g, stencil_set, i, σ, j) # TBD: Should this closure be implemented outside this function?
    ∂∂_narrow(i,σ,j) = mixed_second_derivative_variable_narrow(g, stencil_set, i, σ, j) # TBD: Should this closure be implemented outside this function?

    δ(i,j) = dirac_delta(size(g), i, j)

    return MatrixTensor(N,N) do i, j
        ∂∂_wide(i,λ,j) + ∂∂_narrow(j, μ, i) + δ(i,j)∘Σₖ∂ₖμ∂ₖ
    end
end


function elastic(grid::MappedGrid, λ, μ, stencil_set)
    # Eᵢⱼuⱼ = (∂ᵢλ∂ⱼ + ∂ⱼμ∂ᵢ + ∂ₖμ∂ₖδᵢⱼ) uⱼ
    #
    #       = J⁻¹(∂̃ₖλJg̃ᵏⁿⁱʲ∂̃ₙ + ∂̃ₖμJg̃ᵏⁿʲⁱ∂̃ₙ + δᵢⱼ∂̃ₖμJg̃ᵏⁿˢˢ∂̃ₙ) uⱼ
    #
    #       = J⁻¹(∂̃ₖλJg̃ᵏⁿⁱʲ∂̃ₙ + ∂̃ₖμJg̃ᵏⁿʲⁱ∂̃ₙ + δᵢⱼ∂̃ₖμJgᵏⁿ∂̃ₙ) uⱼ
    #
    #       = J⁻¹(∂̃ₖλJg̃ᵏⁿⁱʲ∂̃ₙ + ∂̃ₖμJg̃ᵏⁿʲⁱ∂̃ₙ + δᵢⱼ∂̃ₖμJgᵏⁿ∂̃ₙ) uⱼ
    # where g̃ᵏⁿⁱʲ = ∂ξₖ/∂xᵢ ∂ξₙ/∂xⱼ
    # and gᵏⁿ = ∂ξₖ/∂xₛ ∂ξₙ/∂xₛ
    N = ndims(grid)

    ∂ξ∂x = map(inv, jacobian(grid))
    g = map(inv, metric_tensor(grid))
    J = map(det, jacobian(grid))
    J⁻¹ = map(inv, J)

    g̃ = map(CartesianIndices((N,N,N,N))) do I
        k,n,i,j = Tuple(I)

        ∂ξₖ∂xᵢ = componentview(∂ξ∂x,k,i)
        ∂ξₙ∂xⱼ = componentview(∂ξ∂x,n,j)

        ∂ξₖ∂xᵢ*̃∂ξₙ∂xⱼ
    end

    g = map(CartesianIndices((N,N))) do I
        k,n = Tuple(I)
        componentview(g,k,n)
    end

    J̲⁻¹ = DiagonalTensor(J⁻¹)

    ∂̃∂̃_wide(i,σ,j) = mixed_second_derivative_variable_wide(logical_grid(grid),stencil_set,i,σ,j)
    ∂̃∂̃_narrow(i,σ,j) = mixed_second_derivative_variable_narrow(logical_grid(grid),stencil_set,i,σ,j)

    δ(i,j) = dirac_delta(size(grid), i, j)

    return MatrixTensor(N, N) do i,j
        sum(1:N) do k
            sum(1:N) do n
                ∂̃ₖλJgᵏⁿⁱʲ∂̃ₙ = ∂̃∂̃_wide(k, λ*̃J*̃g̃[k,n,i,j], n)

                ∂̃ₖμJgᵏⁿʲⁱ∂̃ₙ = ∂̃∂̃_narrow(k, μ*̃J*̃g̃[k,n,j,i], n)

                δᵢⱼ∂̃ₖμJgᵏⁿˢˢ∂̃ₙ = δ(i,j)∘∂̃∂̃_narrow(k,μ*̃J*̃g[k,n],n)

                return J̲⁻¹∘(∂̃ₖλJgᵏⁿⁱʲ∂̃ₙ + ∂̃ₖμJgᵏⁿʲⁱ∂̃ₙ + δᵢⱼ∂̃ₖμJgᵏⁿˢˢ∂̃ₙ)
            end
        end
    end
end



# Helpers
# =======
"""
    mixed_second_derivative_variable_wide(g::Grid, stencil_set, i, σ, j)

The mixed derivative operator ∂ᵢσ∂ⱼ on `g` as a `LazyTensor` approximated
using first derivative operators from `stencil_set`.

See also: [`first_derivative`](@ref)
"""
function mixed_second_derivative_variable_wide(g::Grid, stencil_set, i, σ, j) # TBD: Should it have mixed in the name? It's not mixed when i==j.
    ∂(i) = first_derivative(g, stencil_set, i)

    return ∂(i)∘DiagonalTensor(σ)∘∂(j)
end


"""
    mixed_second_derivative_variable_narrow(g::Grid, stencil_set, i, σ, j)

The mixed derivative operator ∂ᵢσ∂ⱼ on `g` as a `LazyTensor` approximated
using operators from from `stencil_set`. When i = j narrow variable coefficient
second derivative operators are used, while for i  ≠ j first derivative operators
are used.

See also: [`first_derivative`](@ref), [`second_derivative_variable`](@ref)
"""
function mixed_second_derivative_variable_narrow(g::Grid, stencil_set, i, σ, j) # TBD: Should it have mixed in the name? It's not mixed when i==j. Could all of these be MDed under second_derivative?
    ∂(i) = first_derivative(g, stencil_set, i)
    ∂²(σ,i) = second_derivative_variable(g, σ, stencil_set, i)

    if i == j
        ∂²(σ,i)
    else
        ∂(i)∘DiagonalTensor(σ)∘∂(j)
    end
end

# TODO: Can the elastic operators be combined for TensorGrid and MappedGrid?
