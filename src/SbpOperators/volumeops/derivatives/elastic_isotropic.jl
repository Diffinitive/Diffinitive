# Elastic wave equation:
# ρüᵢ = ∂ᵢλ∂ⱼuⱼ + ∂ⱼμ∂ᵢuⱼ + ∂ₖμ∂ₖuᵢ
#     = (∂ᵢλ∂ⱼ + ∂ⱼμ∂ᵢ + ∂ₖμ∂ₖδᵢⱼ) uⱼ

# Traction:
# tᵢ = (nᵢλ∂ⱼ + nⱼμ∂ᵢ + nₖμ∂ₖδᵢⱼ) uⱼ

# Normal traction operator:
# tₙ = nᵢtᵢ = (λ∂ⱼ + 2μnⱼnᵢ∂ᵢ) uⱼ
#           = (λ∂ⱼ + 2μnⱼ∂ₙ) uⱼ

# Tangential traction operator:
# tₜ = tᵢ - nₖtₖnᵢ = μ(nⱼ∂ᵢ + (δᵢⱼ - 2nᵢnⱼ)nₖ∂ₖ) uⱼ
#                  = μ(nⱼ∂ᵢ + (δᵢⱼ - 2nᵢnⱼ)∂ₙ) uⱼ

# for 2d we have
# v₁ = ∂₁λ∂₁u₁ + ∂₁λ∂₂u₂ +
#      ∂₁μ∂₁u₁ + ∂₂μ∂₁u₂ +
#      ∂₁μ∂₁u₁ + ∂₂μ∂₂u₁
# v₂ = ∂₂λ∂₁u₁ + ∂₂λ∂₂u₂ +
#      ∂₁μ∂₂u₁ + ∂₂μ∂₂u₂ +
#      ∂₁μ∂₁u₂ + ∂₂μ∂₂u₂


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

    ∂(i) = first_derivative(g, stencil_set, i)
    ∂²(σ,i) = second_derivative_variable(g, σ, stencil_set, i)

    ∂∂_wide(i,σ,j) = ∂(i)∘DiagonalTensor(σ)∘∂(j)
    ∂∂_narrow(i,σ,j) = i==j ? ∂²(σ,i) : ∂(i)∘DiagonalTensor(σ)∘∂(j)

    δ(i,j) = i==j ? IdentityTensor(size(g)) : ZeroTensor(size(g))

    return MatrixTensor(N,N) do i, j
        ∂∂_wide(i,λ,j) + ∂∂_narrow(j, μ, i) + δ(i,j)∘Σₖ∂ₖμ∂ₖ
    end
end


function traction_isotropic(g::TensorGrid, λ, μ, stencil_set, boundary)
    # nᵢλ∂ⱼuⱼ + nⱼμ∂ᵢuⱼ + nₖμ∂ₖuᵢ
    # =>
    # nᵢλ∂ⱼuⱼ + nⱼμ∂ᵢuⱼ + nₖμ∂ₖδᵢⱼuⱼ
    # (nᵢλ∂ⱼ + nⱼμ∂ᵢ + nₖμ∂ₖδᵢⱼ) uⱼ

    N = ndims(g)

    e = boundary_restriction(g, stencil_set, boundary)

    λ̲ = DiagonalTensor(e*λ)
    μ̲ = DiagonalTensor(e*μ)
    Σₖnₖμ∂ₖ = μ̲∘normal_derivative(g, stencil_set, boundary)

    n = normal(g, boundary)

    ∇ = boundary_gradient(g, stencil_set, boundary)


    n̲(i) = DiagonalTensor(componentview(n,i))

    ∂(i) = e∘first_derivative(g, stencil_set, i)

    bg = boundary_grid(g, boundary)
    δ(i,j) = i==j ? IdentityTensor(size(bg)) : ZeroTensor(size(bg))

    return MatrixTensor(N,N) do i, j
        n̲(i)∘λ̲∘∂(j) + n̲(j)∘μ̲∘∇[i] + δ(i,j)∘Σₖnₖμ∂ₖ
    end
end


function normal_traction_isotropoic(g::TensorGrid, λ, μ, stencil_set, boundary)
    # tₙ = nᵢtᵢ = (λ∂ⱼ + 2μnⱼ∂ₙ) uⱼ

    e = boundary_restriction(g, stencil_set, boundary)

    λ̲ = DiagonalTensor(e*λ)
    μ̲ = DiagonalTensor(e*μ)

    n̲(i) = DiagonalTensor(componentview(n,i))

    ∂(i) = e∘first_derivative(g, stencil_set, i)

    ∂ₙ = normal_derivative(g, stencil_set, boundary)
    return VectorDot(ndims(g)) do j
        λ̲∘∂(j) + 2μ̲∘n̲(j)∘∂ₙ
    end
end

function tangential_traction_isotropic(g::TensorGrid, λ, μ, stencil_set, boundary)
    # tₜ = μ(nⱼ∂ᵢ + (δᵢⱼ - 2nᵢnⱼ)∂ₙ) uⱼ


    e = boundary_restriction(g, stencil_set, boundary)

    λ̲ = DiagonalTensor(e*λ)
    μ̲ = DiagonalTensor(e*μ)

    n̲(i) = DiagonalTensor(componentview(n,i))

    ∂(i) = e∘first_derivative(g, stencil_set, i)

    ∂ₙ = normal_derivative(g, stencil_set, boundary)

    bg = boundary_grid(g, boundary)
    δ(i,j) = i==j ? IdentityTensor(size(bg)) : ZeroTensor(size(bg))

    return MatrixTensor(ndims(g),ndims(g)) do i, j
        μ̲∘(n̲(j)∘∂(i) + (δ(i,j) - 2n̲(i)∘n̲(j))∘∂ₙ)
    end
end

# Mapped grid
# ===========
function elastic_isotropic(grid::MappedGrid, λ, μ, stencil_set)
    # Lᵢⱼuⱼ = (∂ᵢλ∂ⱼ + ∂ⱼμ∂ᵢ + ∂ₖμ∂ₖδᵢⱼ) uⱼ
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

    ∂̃(i) = first_derivative(logical_grid(grid), stencil_set, i)
    ∂̃²(σ,i) = second_derivative_variable(logical_grid(grid), σ, stencil_set, i)

    ∂̃∂̃_wide(i,σ,j) = ∂̃(i)∘DiagonalTensor(σ)∘∂̃(j)
    ∂̃∂̃_narrow(i,σ,j) = i==j ? ∂̃²(σ,i) : ∂̃(i)∘DiagonalTensor(σ)∘∂̃(j)

    δ(i,j) = i==j ? IdentityTensor(size(grid)) : ZeroTensor(size(grid))

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


function traction_isotropic(g::MappedGrid, λ, μ, stencil_set, boundary)
    # In standard coordinates:
    # Tᵢ = nᵢλ∂ⱼuⱼ + nⱼμ∂ᵢuⱼ + nₖμ∂ₖuᵢ
    # nᵢλ∂ⱼuⱼ + nⱼμ∂ᵢuⱼ + nₖμ∂ₖδᵢⱼuⱼ
    # (nᵢλ∂ⱼ + nⱼμ∂ᵢ + nₖμ∂ₖδᵢⱼ) uⱼ
    #
    #  With fᵢⱼ = ∂ξᵢ/∂xⱼ => ∂ᵢ = fₖᵢ∂̃ₖ
    #  we have
    #
    # Tᵢ = (nᵢλfₖⱼ∂̃ₖ + nⱼμfₖᵢ∂̃ₖ + nₛμfₖₛ∂̃ₖδᵢⱼ) uⱼ
    #

    N = ndims(g)


    e = boundary_restriction(g, stencil_set, boundary)
    ∂ξ∂x = collect(e*map(inv, jacobian(g)))

    f̲ = [DiagonalTensor(componentview(∂ξ∂x, i, j)) for i∈1:N, j∈1:N]

    λ̲ = DiagonalTensor(e*λ)
    μ̲ = DiagonalTensor(e*μ)

    n = normal(g, boundary)

    nf = map(n, ∂ξ∂x) do n, f
        f*n
    end

    n̲f̲ = [DiagonalTensor(componentview(nf, i)) for i ∈ 1:N]

    ∇̃ = boundary_gradient(logical_grid(g), stencil_set, boundary)
    n̲(i) = DiagonalTensor(componentview(n,i))
    ∂̃(i) = e∘first_derivative(logical_grid(g), stencil_set, i)

    bg = boundary_grid(g, boundary)
    δ(i,j) = i==j ? IdentityTensor(size(bg)) : ZeroTensor(size(bg))

    return MatrixTensor(N,N) do i, j
        sum(1:N) do k
            n̲(i)∘λ̲∘f̲[k,j]∘∂̃(k) + n̲(j)∘μ̲∘f̲[k,i]∘∇̃[k] + δ(i,j)∘μ̲∘n̲f̲[k]∘∇̃[k]
        end
    end
end


function normal_traction_isotropoic(g::MappedGrid, λ, μ, stencil_set, boundary)
    # tₙ = nᵢtᵢ = (λ∂ⱼ + 2μnⱼ∂ₙ) uⱼ
    #
    # With fᵢⱼ = ∂ξᵢ/∂xⱼ => ∂ᵢ = fₖᵢ∂̃ₖ
    # we have
    #
    # tₙ = (λfₖⱼ∂̃ₖ + 2μnⱼ∂ₙ) uⱼ


    e = boundary_restriction(g, stencil_set, boundary)

    ∂ξ∂x = collect(e*map(inv, jacobian(g)))

    f̲ = [DiagonalTensor(componentview(∂ξ∂x, i, j)) for i∈1:N, j∈1:N]

    λ̲ = DiagonalTensor(e*λ)
    μ̲ = DiagonalTensor(e*μ)

    n̲(i) = DiagonalTensor(componentview(n,i))

    ∂̃(i) = e∘first_derivative(g, stencil_set, i)

    ∂ₙ = normal_derivative(g, stencil_set, boundary)

    N = ndims(g)
    return VectorDot(N) do j
        fₖⱼ∂̃ₖ = sum(k->f[k,j]∘∂̃(k), 1:N)
        λ̲∘fₖⱼ∂̃ₖ + 2μ̲∘n̲(j)∘∂ₙ
    end
end

function tangential_traction_isotropic(g::MappedGrid, λ, μ, stencil_set, boundary)
    # tₜ = μ(nⱼ∂ᵢ + (δᵢⱼ - 2nᵢnⱼ)∂ₙ) uⱼ
    #
    # With fᵢⱼ = ∂ξᵢ/∂xⱼ => ∂ᵢ = fₖᵢ∂̃ₖ
    # we have
    #
    # tₜ = μ(nⱼfₖᵢ∂̃ₖ + (δᵢⱼ - 2nᵢnⱼ)∂ₙ) uⱼ


    e = boundary_restriction(g, stencil_set, boundary)

    ∂ξ∂x = collect(e*map(inv, jacobian(g)))

    f̲ = [DiagonalTensor(componentview(∂ξ∂x, i, j)) for i∈1:N, j∈1:N]

    λ̲ = DiagonalTensor(e*λ)
    μ̲ = DiagonalTensor(e*μ)

    n̲(i) = DiagonalTensor(componentview(n,i))

    ∂̃(i) = e∘first_derivative(g, stencil_set, i)

    ∂ₙ = normal_derivative(g, stencil_set, boundary)

    bg = boundary_grid(g, boundary)
    δ(i,j) = i==j ? IdentityTensor(size(bg)) : ZeroTensor(size(bg))

    return MatrixTensor(ndims(g),ndims(g)) do i, j
        fₖᵢ∂̃ₖ = sum(k->f[k,i]∘∂̃(k), 1:N)
        μ̲∘(n̲(j)∘fₖᵢ∂̃ₖ + (δ(i,j) - 2n̲(i)∘n̲(j))∘∂ₙ)
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



