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

    _∂∂_wide(i,σ,j) = ∂∂_wide(g, stencil_set, i, σ, j) # TBD: Should this closure be implemented outside this function?
    _∂∂_narrow(i,σ,j) = ∂∂_narrow(g, stencil_set, i, σ, j) # TBD: Should this closure be implemented outside this function?

    _δ(i,j) = δ(g, i, j)

    return MatrixTensor(N,N) do i, j
        _∂∂_wide(i,λ,j) + _∂∂_narrow(j, μ, i) + _δ(i,j)∘Σₖ∂ₖμ∂ₖ
    end
end


function traction_isotropic(g::TensorGrid, λ, μ, stencil_set, boundary)
    # nᵢλ∂ⱼuⱼ + nⱼμ∂ᵢuⱼ + nₖμ∂ₖuᵢ
    # =>
    # nᵢλ∂ⱼuⱼ + nⱼμ∂ᵢuⱼ + nₖμ∂ₖδᵢⱼuⱼ
    # (nᵢλ∂ⱼ + nⱼμ∂ᵢ + nₖμ∂ₖδᵢⱼ) uⱼ
    # (nᵢλ∂ⱼ + nⱼμ∂ᵢ + δᵢⱼμ∂ₙ) uⱼ

    N = ndims(g)

    e = boundary_restriction(g, stencil_set, boundary)
    λ̲ = DiagonalTensor(e*λ)
    μ̲ = DiagonalTensor(e*μ)

    n = normal(g, boundary)
    n̲(i) = DiagonalTensor(componentview(n,i))

    ∂_wide(i) = ∂_wide(g, stencil_set, boundary, i)
    ∂_narrow(i) = ∂_narrow(g, stencil_set, boundary, i)
    δ(i,j) = δ(g, boundary, i, j)
    ∂ₙ = normal_derivative(g, stencil_set, boundary)

    return MatrixTensor(N,N) do i, j
        n̲(i)∘λ̲∘∂_wide(j) + n̲(j)∘μ̲∘∂_narrow(i) + δ(i,j)∘μ̲∘∂ₙ
    end
end


function normal_traction_isotropoic(g::TensorGrid, λ, μ, stencil_set, boundary)
    # tₙ = nᵢtᵢ = (λ∂ⱼ + 2μnⱼ∂ₙ) uⱼ

    e = boundary_restriction(g, stencil_set, boundary)

    λ̲ = DiagonalTensor(e*λ)
    μ̲ = DiagonalTensor(e*μ)

    n = normal(g, boundary)
    n̲(i) = DiagonalTensor(componentview(n,i))

    ∂(i) = ∂_wide(g, stencil_set, boundary, i)

    ∂ₙ = normal_derivative(g, stencil_set, boundary)
    return VectorDotTensor(ndims(g)) do j
        λ̲∘∂(j) + 2μ̲∘n̲(j)∘∂ₙ
    end
end

function tangential_traction_isotropic(g::TensorGrid, λ, μ, stencil_set, boundary)
    # tₜ = μ(nⱼ∂ᵢ + (δᵢⱼ - 2nᵢnⱼ)∂ₙ) uⱼ

    e = boundary_restriction(g, stencil_set, boundary)

    μ̲ = DiagonalTensor(e*μ)

    n = normal(g, boundary)
    n̲(i) = DiagonalTensor(componentview(n,i))

    ∂(i) = ∂_narrow(g, stencil_set, boundary, i)
    ∂ₙ = normal_derivative(g, stencil_set, boundary)

    _δ(i,j) = δ(g, boundary, i, j)

    return MatrixTensor(ndims(g),ndims(g)) do i, j
        μ̲∘(n̲(j)∘∂(i) + (_δ(i,j) - 2n̲(i)∘n̲(j))∘∂ₙ)
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

    ∂̃∂̃_wide(i,σ,j) = ∂∂_wide(logical_grid(grid),stencil_set,i,σ,j)
    ∂̃∂̃_narrow(i,σ,j) = ∂∂_narrow(logical_grid(grid),stencil_set,i,σ,j)

    _δ(i,j) = δ(grid, i, j)

    return MatrixTensor(N, N) do i,j
        sum(1:N) do k
            sum(1:N) do n
                ∂̃ₖλJgᵏⁿⁱʲ∂̃ₙ = ∂̃∂̃_wide(k, λ*̃J*̃g̃[k,n,i,j], n)

                ∂̃ₖμJgᵏⁿʲⁱ∂̃ₙ = ∂̃∂̃_narrow(k, μ*̃J*̃g̃[k,n,j,i], n)

                δᵢⱼ∂̃ₖμJgᵏⁿˢˢ∂̃ₙ = _δ(i,j)∘∂̃∂̃_narrow(k,μ*̃J*̃g[k,n],n)

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

    sn̲(i) = DiagonalTensor(componentview(n,i))

    ∂̃_wide(i) = ∂_wide(logical_grid(g), stencil_set, boundary, i)
    ∂̃_narrow(i) = ∂_narrow(logical_grid(g), stencil_set, boundary, i)

    _δ(i,j) = δ(g, boundary, i, j)

    return MatrixTensor(N,N) do i, j
        sum(1:N) do k
            n̲(i)∘λ̲∘f̲[k,j]∘∂̃_wide(k) + n̲(j)∘μ̲∘f̲[k,i]∘∂̃_narrow(k) + _δ(i,j)∘μ̲∘n̲f̲[k]∘∂̃_narrow(k)
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

    n = normal(g, boundary)
    n̲(i) = DiagonalTensor(componentview(n,i))

    ∂̃(i) = ∂_wide(g, stencil_set, boundary, i)

    ∂ₙ = normal_derivative(g, stencil_set, boundary)

    N = ndims(g)
    return VectorDotTensor(N) do j
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

    N = ndims(g)
    e = boundary_restriction(g, stencil_set, boundary)

    ∂ξ∂x = collect(e*map(inv, jacobian(g)))

    f̲ = [DiagonalTensor(componentview(∂ξ∂x, i, j)) for i∈1:N, j∈1:N]

    μ̲ = DiagonalTensor(e*μ)

    n = normal(g, boundary)
    n̲(i) = DiagonalTensor(componentview(n,i))

    ∂̃(i) = ∂_narrow(g, stencil_set, boundary, i)

    ∂ₙ = normal_derivative(g, stencil_set, boundary)

    _δ(i,j) = δ(g, boundary, i, j)

    return MatrixTensor(ndims(g),ndims(g)) do i, j
        fₖᵢ∂̃ₖ = sum(k->f[k,i]∘∂̃(k), 1:N)
        μ̲∘(n̲(j)∘fₖᵢ∂̃ₖ + (_δ(i,j) - 2n̲(i)∘n̲(j))∘∂ₙ)
    end
end


# Helpers
# =======
function boundary_gradient(g, stencil_set, boundary)
    return map(1:ndims(g)) do i
        if i == grid_id(boundary)
            s = Grids._boundary_sign(component_type(g), boundary)
            ∂ₙ = normal_derivative(g, stencil_set, boundary)
            return s*∂ₙ
        else
            e = boundary_restriction(g, stencil_set, boundary)
            ∂ᵢ = first_derivative(g, stencil_set, i)
            return e∘∂ᵢ
        end
    end
end



∂(g::Grid, stencil_set, i) = first_derivative(g, stencil_set, i)
∂²(g::Grid, stencil_set, σ,i) = second_derivative_variable(g, σ, stencil_set, i)


function ∂∂_wide(g::Grid, stencil_set, i, σ, j)
    _∂(i) = ∂(g, stencil_set, i) # TBD: Should this closure be implemented outside this function?

    return _∂(i)∘DiagonalTensor(σ)∘_∂(j)
end

function ∂∂_narrow(g::Grid, stencil_set, i, σ, j)
    _∂(i) = ∂(g, stencil_set, i) # TBD: Should this closure be implemented outside this function?
    _∂²(σ,i) = ∂²(g, stencil_set, σ, i) # TBD: Should this closure be implemented outside this function?

    if i == j
        _∂²(σ,i)
    else
        _∂(i)∘DiagonalTensor(σ)∘_∂(j)
    end
end

function ∂_wide(g::Grid, stencil_set, boundary::BoundaryIdentifier, i)
    e = boundary_restriction(g, stencil_set, boundary)
    ∂ᵢ = ∂(g, stencil_set, i)

    return e∘∂ᵢ
end

function ∂_narrow(g::Grid, stencil_set, boundary::BoundaryIdentifier, i)
    if i == grid_id(boundary)
        s = Grids._boundary_sign(component_type(g), boundary)
        ∂ₙ = normal_derivative(g, stencil_set, boundary)
        return s*∂ₙ
    else
        e = boundary_restriction(g, stencil_set, boundary)
        ∂ᵢ = first_derivative(g, stencil_set, i)
        return e∘∂ᵢ
    end
end

# TODO: Extend the two functions above for mapped grids
# TODO: Move the two functions above to a more general spot?

function δ(g::Grid, i, j)
    if i==j
        IdentityTensor(size(g))
    else
        ZeroTensor(size(g))
    end
end

function δ(g::Grid, boundary::BoundaryIdentifier, i, j)
    bg = boundary_grid(g, boundary)
    return δ(bg, i, j)
end


# TODO: Can the traction operators be combined for TensorGrid and MappedGrid?
# TODO: Can the elastic operators be combined for TensorGrid and MappedGrid?
# TODO: Would it be helpful to implement operators for getting the normal and tangential projections of vectors on the boundary?
#       Could these be used to simplify the implementations of normal traction and tangential traction?
#       (Yes? Simplest would be to construct them on the boundary grid. Then they could be combined with the regular Traction operator to get the components.)
