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
