function elastic_isotropic(g::TensorGrid, λ, μ, stencil_set)
    # ∂ᵢλ∂ⱼuⱼ + ∂ⱼμ∂ᵢuⱼ + ∂ₖμ∂ₖuᵢ
    # =>
    # ∂ᵢλ∂ⱼuⱼ + ∂ⱼμ∂ᵢuⱼ + ∂ₖμ∂ₖδᵢⱼuⱼ
    # (∂ᵢλ∂ⱼ + ∂ⱼμ∂ᵢ + ∂ₖμ∂ₖδᵢⱼ) uⱼ

    N = ndims(g)

    ∂ₖμ∂ₖ = sum(1:N) do k
        second_derivative_variable(g,μ,stencil_set,k)
    end

    Λ = DiagonalTensor(λ)
    M = DiagonalTensor(μ)


    Ds = map(Iterators.product(1:N, 1:N)) do (i,j)
        ∂ᵢ = first_derivative(g, stencil_set, i)
        ∂ⱼ = first_derivative(g, stencil_set, j)
        if i == j
            ∂ⱼμ∂ᵢ = second_derivative_variable(g,μ,stencil_set,i)
            # ∂ₖμ∂ₖδᵢⱼ = ∂ₖμ∂ₖ
            return ∂ᵢ∘Λ∘∂ⱼ + ∂ⱼμ∂ᵢ + ∂ₖμ∂ₖ
        else
            ∂ⱼμ∂ᵢ = ∂ⱼ∘M∘∂ᵢ
            # ∂ₖμ∂ₖδᵢⱼ = ZeroTensor()
            return ∂ᵢ∘Λ∘∂ⱼ + ∂ⱼμ∂ᵢ
        end

        return ∂ᵢ∘Λ∘∂ⱼ + ∂ⱼμ∂ᵢ + ∂ₖμ∂ₖδᵢⱼ
    end

    return MatrixTensor(Ds)
end

