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
            return nᵢ∘Λ∘∇[j] + nⱼ∘M∘∇[i] + Σₖnₖμ∂ₖ
        else
            return nᵢ∘Λ∘e∘∂ⱼ + nⱼ∘M∘e∘∂ᵢ
        end
    end

    return MatrixTensor(Ds)
end


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
