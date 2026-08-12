# Traction operator:
# Tᵢ = (nᵢλ∂ⱼ + nⱼμ∂ᵢ + nₖμ∂ₖδᵢⱼ) uⱼ

# Normal traction operator:
# Tₙ = nᵢTᵢ = (λ∂ⱼ + 2μnⱼnᵢ∂ᵢ) uⱼ
#           = (λ∂ⱼ + 2μnⱼ∂ₙ) uⱼ

# Tangential traction operator:
# tᵢ = Tᵢ - nₖtₖnᵢ = μ(nⱼ∂ᵢ + (δᵢⱼ - 2nᵢnⱼ)nₖ∂ₖ) uⱼ
#                 = μ(nⱼ∂ᵢ + (δᵢⱼ - 2nᵢnⱼ)∂ₙ) uⱼ

# Tensor grid
# ===========
function traction(g::TensorGrid, λ, μ, stencil_set, boundary)
    # nᵢλ∂ⱼuⱼ + nⱼμ∂ᵢuⱼ + nₖμ∂ₖuᵢ
    # =>
    # nᵢλ∂ⱼuⱼ + nⱼμ∂ᵢuⱼ + nₖμ∂ₖδᵢⱼuⱼ
    # (nᵢλ∂ⱼ + nⱼμ∂ᵢ + nₖμ∂ₖδᵢⱼ) uⱼ
    # (nᵢλ∂ⱼ + nⱼμ∂ᵢ + δᵢⱼμ∂ₙ) uⱼ

    N = ndims(g)

    e = boundary_restriction(g, stencil_set, boundary)
    λ̲ = DiagonalTensor(e*λ)
    μ̲ = DiagonalTensor(e*μ)

    n = boundary_normal(g, boundary)
    n̲(i) = DiagonalTensor(componentview(n,i))

    ∂_wide(i) = first_derivative_wide(g, stencil_set, boundary, i)
    ∂_narrow(i) = first_derivative_narrow(g, stencil_set, boundary, i)
    δ(i,j) = dirac_delta(size(boundary_grid(g, boundary)), i, j)
    ∂ₙ = normal_derivative(g, stencil_set, boundary)

    return MatrixTensor(N,N) do i, j
        n̲(i)∘λ̲∘∂_wide(j) + n̲(j)∘μ̲∘∂_narrow(i) + δ(i,j)∘μ̲∘∂ₙ
    end
end


function normal_traction(g::TensorGrid, λ, μ, stencil_set, boundary)
    # tₙ = nᵢtᵢ = (λ∂ⱼ + 2μnⱼ∂ₙ) uⱼ

    e = boundary_restriction(g, stencil_set, boundary)

    λ̲ = DiagonalTensor(e*λ)
    μ̲ = DiagonalTensor(e*μ)

    n = boundary_normal(g, boundary)
    n̲(i) = DiagonalTensor(componentview(n,i))

    ∂(i) = first_derivative_wide(g, stencil_set, boundary, i)

    ∂ₙ = normal_derivative(g, stencil_set, boundary)
    return VectorDotTensor(ndims(g)) do j
        λ̲∘∂(j) + 2μ̲∘n̲(j)∘∂ₙ
    end
end

function tangential_traction(g::TensorGrid, λ, μ, stencil_set, boundary)
    # tₜ = μ(nⱼ∂ᵢ + (δᵢⱼ - 2nᵢnⱼ)∂ₙ) uⱼ

    e = boundary_restriction(g, stencil_set, boundary)

    μ̲ = DiagonalTensor(e*μ)

    n = boundary_normal(g, boundary)
    n̲(i) = DiagonalTensor(componentview(n,i))

    ∂(i) = first_derivative_narrow(g, stencil_set, boundary, i)
    ∂ₙ = normal_derivative(g, stencil_set, boundary)

    δ(i,j) = dirac_delta(size(boundary_grid(g, boundary)), i, j)

    return MatrixTensor(ndims(g),ndims(g)) do i, j
        μ̲∘(n̲(j)∘∂(i) + (δ(i,j) - 2n̲(i)∘n̲(j))∘∂ₙ)
    end
end


# Mapped grid
# ===========
function traction(g::MappedGrid, λ, μ, stencil_set, boundary)
    # In standard coordinates:
    # Tᵢ = nᵢλ∂ⱼuⱼ + nⱼμ∂ᵢuⱼ + nₖμ∂ₖuᵢ
    # nᵢλ∂ⱼuⱼ + nⱼμ∂ᵢuⱼ + nₖμ∂ₖδᵢⱼuⱼ
    # (nᵢλ∂ⱼ + nⱼμ∂ᵢ + nₖμ∂ₖδᵢⱼ) uⱼ
    #
    # With fᵢⱼ = ∂ξᵢ/∂xⱼ => ∂ᵢ = fₖᵢ∂̃ₖ we can write the traction in
    # logical coordinates, we have
    #
    # Tᵢ = (nᵢλfₖⱼ∂̃ₖ + nⱼμfₖᵢ∂̃ₖ + nₛμfₖₛ∂̃ₖδᵢⱼ) uⱼ
    #

    N = ndims(g)

    # Construct parts needed for the tranction operator:
    e = boundary_restriction(g, stencil_set, boundary)
    ∂ξ∂x = collect(e*map(inv, jacobian(g)))

    λ̲ = DiagonalTensor(e*λ)
    μ̲ = DiagonalTensor(e*μ)

    n = boundary_normal(g, boundary)

    nf = map(n, ∂ξ∂x) do n, f
        f*n
    end

    f̲(i,j) = DiagonalTensor(componentview(∂ξ∂x, i, j))
    n̲(i) = DiagonalTensor(componentview(n,i))
    n̲f̲(i) = DiagonalTensor(componentview(nf, i))


    ∂̃_wide(i) = first_derivative_wide(logical_grid(g), stencil_set, boundary, i)
    ∂̃_narrow(i) = first_derivative_narrow(logical_grid(g), stencil_set, boundary, i)

    δ(i,j) = dirac_delta(size(boundary_grid(g, boundary)), i, j)

    # Assemble traction operator:
    return MatrixTensor(N,N) do i, j
        sum(1:N) do k
            n̲(i)∘λ̲∘f̲(k,j)∘∂̃_wide(k) + n̲(j)∘μ̲∘f̲(k,i)∘∂̃_narrow(k) + δ(i,j)∘μ̲∘n̲f̲(k)∘∂̃_narrow(k)
        end
    end
end


function normal_traction(g::MappedGrid, λ, μ, stencil_set, boundary)
    # tₙ = nᵢtᵢ = (λ∂ⱼ + 2μnⱼ∂ₙ) uⱼ
    #
    # With fᵢⱼ = ∂ξᵢ/∂xⱼ => ∂ᵢ = fₖᵢ∂̃ₖ
    # we have
    #
    # tₙ = (λfₖⱼ∂̃ₖ + 2μnⱼ∂ₙ) uⱼ

    e = boundary_restriction(g, stencil_set, boundary)
    ∂ξ∂x = collect(e*map(inv, jacobian(g)))

    λ̲ = DiagonalTensor(e*λ)
    μ̲ = DiagonalTensor(e*μ)

    n = boundary_normal(g, boundary)

    f̲(i,j) = DiagonalTensor(componentview(∂ξ∂x, i, j))
    n̲(i) = DiagonalTensor(componentview(n,i))
    ∂̃(i) = first_derivative_wide(logical_grid(g), stencil_set, boundary, i)

    ∂ₙ = normal_derivative(g, stencil_set, boundary)

    N = ndims(g)
    return VectorDotTensor(N) do j
        fₖⱼ∂̃ₖ = sum(k->f̲(k,j)∘∂̃(k), 1:N)
        λ̲∘fₖⱼ∂̃ₖ + 2μ̲∘n̲(j)∘∂ₙ
    end
end


function tangential_traction(g::MappedGrid, λ, μ, stencil_set, boundary) # TBD: Should we remove dependence on λ here? Add error hint?
    # tₜ = μ(nⱼ∂ᵢ + (δᵢⱼ - 2nᵢnⱼ)∂ₙ) uⱼ
    #
    # With fᵢⱼ = ∂ξᵢ/∂xⱼ => ∂ᵢ = fₖᵢ∂̃ₖ
    # we have
    #
    # tₜ = μ(nⱼfₖᵢ∂̃ₖ + (δᵢⱼ - 2nᵢnⱼ)∂ₙ) uⱼ

    e = boundary_restriction(g, stencil_set, boundary)
    ∂ξ∂x = collect(e*map(inv, jacobian(g)))

    μ̲ = DiagonalTensor(e*μ)

    n = boundary_normal(g, boundary)

    f̲(i,j) = DiagonalTensor(componentview(∂ξ∂x, i, j))
    n̲(i) = DiagonalTensor(componentview(n,i))
    ∂̃(i) = first_derivative_narrow(logical_grid(g), stencil_set, boundary, i)

    ∂ₙ = normal_derivative(g, stencil_set, boundary)

    δ(i,j) = dirac_delta(size(boundary_grid(g,boundary)), i, j)

    N = ndims(g)
    return MatrixTensor(ndims(g),ndims(g)) do i, j
        fₖᵢ∂̃ₖ = sum(k->f̲(k,i)∘∂̃(k), 1:N)
        μ̲∘(n̲(j)∘fₖᵢ∂̃ₖ + (δ(i,j) - 2n̲(i)∘n̲(j))∘∂ₙ)
    end
end

# Helpers
# =======
# TODO: Not used? Do we need it?
function boundary_gradient(g, stencil_set, boundary) # Does this need to have "boundary" in the name?
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


# wide and narrow first derviatives only relevant to the boundary
function first_derivative_wide(g::Grid, stencil_set, boundary::BoundaryIdentifier, i)
    e = boundary_restriction(g, stencil_set, boundary)
    ∂ᵢ = first_derivative(g, stencil_set, i)

    return e∘∂ᵢ
end

function first_derivative_narrow(g::Grid, stencil_set, boundary::BoundaryIdentifier, i)
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
# TODO: Can the traction operators be combined for TensorGrid and MappedGrid?
# TODO: Would it be helpful to implement operators for getting the normal and tangential projections of vectors on the boundary?
#       Could these be used to simplify the implementations of normal traction and tangential traction?
#       (Yes? Simplest would be to construct them on the boundary grid. Then they could be combined with the regular Traction operator to get the components.)
