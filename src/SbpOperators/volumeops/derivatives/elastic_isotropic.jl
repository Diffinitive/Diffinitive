# Tensor grid
# ===========
function elastic_isotropic(g::TensorGrid, λ, μ, stencil_set)
    # ∂ᵢλ∂ⱼuⱼ + ∂ⱼμ∂ᵢuⱼ + ∂ₖμ∂ₖuᵢ
    # =>
    # ∂ᵢλ∂ⱼuⱼ + ∂ⱼμ∂ᵢuⱼ + ∂ₖμ∂ₖδᵢⱼuⱼ
    # (∂ᵢλ∂ⱼ + ∂ⱼμ∂ᵢ + ∂ₖμ∂ₖδᵢⱼ) uⱼ

    N = ndims(g)

    Σₖ∂ₖμ∂ₖ = sum(1:N) do k
        second_derivative_variable(g,μ,stencil_set,k)
    end

    Λ = DiagonalTensor(λ)
    M = DiagonalTensor(μ)

    Ds = map(Iterators.product(1:N, 1:N)) do (i,j)
        ∂ᵢ = first_derivative(g, stencil_set, i)
        ∂ⱼ = first_derivative(g, stencil_set, j)
        if i == j
            ∂ⱼμ∂ᵢ = second_derivative_variable(g,μ,stencil_set,i)
            return ∂ᵢ∘Λ∘∂ⱼ + ∂ⱼμ∂ᵢ + Σₖ∂ₖμ∂ₖ
        else
            return ∂ᵢ∘Λ∘∂ⱼ + ∂ⱼ∘M∘∂ᵢ
        end
    end

    return MatrixTensor(Ds)
end


function traction_isotropic(g::TensorGrid, λ, μ, stencil_set, boundary)
    # nᵢλ∂ⱼuⱼ + nⱼμ∂ᵢuⱼ + nₖμ∂ₖuᵢ
    # =>
    # nᵢλ∂ⱼuⱼ + nⱼμ∂ᵢuⱼ + nₖμ∂ₖδᵢⱼuⱼ
    # (nᵢλ∂ⱼ + nⱼμ∂ᵢ + nₖμ∂ₖδᵢⱼ) uⱼ

    N = ndims(g)

    e = boundary_restriction(g, stencil_set, boundary)

    Λ = DiagonalTensor(e*λ)
    M = DiagonalTensor(e*μ)

    n = normal(g, boundary)

    ∇ = boundary_gradient(g, stencil_set, boundary)

    Ds = map(Iterators.product(1:N, 1:N)) do (i,j)
        ∂ᵢ = first_derivative(g, stencil_set, i)
        ∂ⱼ = first_derivative(g, stencil_set, j)

        nᵢ = DiagonalTensor(componentview(n, i))
        nⱼ = DiagonalTensor(componentview(n, j))
        if i == j
            Σₖnₖμ∂ₖ = M∘normal_derivative(g, stencil_set, boundary)
            return nᵢ∘Λ∘e∘∂ⱼ + nⱼ∘M∘∇[i] + Σₖnₖμ∂ₖ
        else
            return nᵢ∘Λ∘e∘∂ⱼ + nⱼ∘M∘e∘∂ᵢ
        end
    end

    return MatrixTensor(Ds)
end


# Mapped grid
# ===========
function elastic_isotropic(g::MappedGrid, λ, μ, stencil_set)
    # Lᵢⱼuⱼ = (∂ᵢλ∂ⱼ + ∂ⱼμ∂ᵢ + ∂ₖμ∂ₖδᵢⱼ) uⱼ
    #
    #       = J⁻¹(∂̃ₖλJg̃ᵏⁿⁱʲ∂̃ₙ + ∂̃ₖμJg̃ᵏⁿʲⁱ∂̃ₙ + δᵢⱼ∂̃ₖμJg̃ᵏⁿˢˢ∂̃ₙ) uⱼ
    #
    #       = J⁻¹(∂̃ₖλJg̃ᵏⁿⁱʲ∂̃ₙ + ∂̃ₖμJg̃ᵏⁿʲⁱ∂̃ₙ + δᵢⱼ∂̃ₖμJgᵏⁿ∂̃ₙ) uⱼ
    # where g̃ᵏⁿⁱʲ = ∂ξₖ/∂xᵢ ∂ξₙ/∂xⱼ
    # and gᵏⁿ = ∂ξₖ/∂xₛ ∂ξₙ/∂xₛ
    N = ndims(g)

    ∂ξ∂x = map(inv, jacobian(g))
    J = map(det, jacobian(g))
    J⁻¹ = map(inv, J)

    λJg̃ = map(CartesianIndices((N,N,N,N))) do I
        k,n,i,j = Tuple(I)

        ∂ξₖ∂xᵢ = componentview(∂ξ∂x,k,i)
        ∂ξₙ∂xⱼ = componentview(∂ξ∂x,n,j)

        λ.*J.*∂ξₖ∂xᵢ.*∂ξₙ∂xⱼ
    end

    g = map(CartesianIndices((N,N))) do I
        a,b = Tuple(I)
    end


    g(a,b,c,d) = nothing


    ∂̃(i) = first_derivative(logical_grid(g), stencil_set, i)
    ∂̃²(σ,i) = second_derivative_variable(logical_grid(g), σ, stencil_set, i)
    ∂̃σ∂̃(i,σ,j) = i==j ? ∂̃²(σ,i) : ∂̃(i)∘DiagonalTensor(σ)∘∂̃(j)

    δ(i,j) = i==j ? IdentityTensor(size(lg)) : ZeroTensor(size(lg))


    return MatrixTensor(N, N) do i,j
        sum(1:N) do k
            sum(1:N) do n
                ∂̃ₖλJgᵏⁿⁱʲ∂̃ₙ = ∂̃(k)∘DiagonalTensor(λJg̃[k,n,i,j])∘∂̃(n)

                ∂̃ₖμJgᵏⁿʲⁱ∂̃ₙ = ∂̃σ∂̃(k, μJg̃[k,n,j,i], n)

                δᵢⱼ∂̃ₖμJgᵏⁿˢˢ∂̃ₙ = δ(i,j)∘∂̃σ∂̃(k,μJg[k,n],n) # TODO: Kolla att δ försvinner helt

                return ∂̃ₖλJgᵏⁿⁱʲ∂̃ₙ + ∂̃ₖμJgᵏⁿʲⁱ∂̃ₙ + δᵢⱼ∂̃ₖμJgᵏⁿˢˢ∂̃ₙ
            end
        end
    end
end



# Helpers
# =======
function boundary_gradient(g, stencil_set, boundary)
    e = boundary_restriction(g, stencil_set, boundary)
    return map(1:ndims(g)) do i
        if i == grid_id(boundary)
            s = Grids._boundary_sign(component_type(g), boundary)
            ∂ₙ = normal_derivative(g, stencil_set, boundary)
            return s*∂ₙ
        else
            ∂ᵢ = first_derivative(g, stencil_set, i)
            return e∘∂ᵢ
        end
    end
end



