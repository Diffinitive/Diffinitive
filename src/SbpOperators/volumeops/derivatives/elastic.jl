# Elastic operator:
# Eᵢⱼuⱼ = ∂ᵢλ∂ⱼuⱼ + ∂ⱼμ∂ᵢuⱼ + ∂ₖμ∂ₖuᵢ
#       = (∂ᵢλ∂ⱼ + ∂ⱼμ∂ᵢ + ∂ₖμ∂ₖδᵢⱼ) uⱼ

# for 2d we have
# v₁ = E₁ⱼuⱼ = ∂₁λ∂₁u₁ + ∂₁λ∂₂u₂ +
#              ∂₁μ∂₁u₁ + ∂₂μ∂₁u₂ +
#              ∂₁μ∂₁u₁ + ∂₂μ∂₂u₁
# v₂ = E₂ⱼuⱼ = ∂₂λ∂₁u₁ + ∂₂λ∂₂u₂ +
#              ∂₁μ∂₂u₁ + ∂₂μ∂₂u₂ +
#              ∂₁μ∂₁u₂ + ∂₂μ∂₂u₂


# Tensor grid
# ===========
function elastic(g::TensorGrid, λ, μ, stencil_set)
    # ∂ᵢλ∂ⱼuⱼ + ∂ⱼμ∂ᵢuⱼ + ∂ₖμ∂ₖuᵢ
    # =>
    # ∂ᵢλ∂ⱼuⱼ + ∂ⱼμ∂ᵢuⱼ + ∂ₖμ∂ₖδᵢⱼuⱼ
    # (∂ᵢλ∂ⱼ + ∂ⱼμ∂ᵢ + ∂ₖμ∂ₖδᵢⱼ) uⱼ

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


# Mapped grid
# ===========
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
# ∂ᵢσ∂ⱼ for given i,σ,j using all D1
function mixed_second_derivative_variable_wide(g::Grid, stencil_set, i, σ, j) # TBD: Should it have mixed in the name? It's not mixed when i==j.
    ∂(i) = first_derivative(g, stencil_set, i)

    return ∂(i)∘DiagonalTensor(σ)∘∂(j)
end


# ∂ᵢσ∂ⱼ for given i,σ,j using D2 when i = j and D1 otherwise
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