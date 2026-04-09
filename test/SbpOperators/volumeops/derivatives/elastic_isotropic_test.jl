using Test

using Diffinitive.SbpOperators
using Diffinitive.Grids
# using Diffinitive.LazyTensors


# SBP-factorization
# Accuracy

# TensorGrid
# MappedGrid

# 2D
# 3D

# Full operator
# Traction
# Normal traction
# Tangential traction




## Automatic differentiation
onehot(k,N) = SVector(ntuple(i->k==i ? 1 : 0,N))
δ(i,j) = i==j ? 1 : 0

e(u, i, x) = u(x)[i]
e(u,i) = x->e(u,i,x)

∂(f,d::AbstractArray,x) = ForwardDiff.derivative(s->f(x+s*d),0)
∂(f,d::AbstractArray) = x->∂(f,d,x)

∂(f,i::Int,x) = ∂(f,onehot(i, length(x)),x)
∂(f,i::Int) = x->∂(f,i,x)

∂∂(f, i, j, x::AbstractArray) = ∂(∂(f,j),i,x)
∂∂(f, i, j) = x->∂∂(f, i, j, x::AbstractArray)

∂∂(f, i, σ, j, x::AbstractArray) = ∂(x->σ(x)*∂(f,j,x),i,x)
∂∂(f, i, σ, j) = x->∂∂(f, i, σ, j, x::AbstractArray)

Δ(f,σ,x) = sum(k->∂∂(f,σ,x), ntuple(identity, length(x)))
Δ(f,σ) = x->Δ(f,σ,x)

div(f, x) = sum(k->∂(f,k,x), ntuple(identity,length(x)))
div(f) = x->div(f,x)


function elastic_ad(u, λ, μ, x)
    map(ntuple(identity, length(x))) do i
        sum(ntuple(identity, length(x)) do j
            uⱼ = e(u,j)
            # ∂ᵢλ∂ⱼuⱼ + ∂ⱼμ∂ᵢuⱼ + δᵢⱼ∂ₖμ∂ₖuⱼ
            ∂∂(uⱼ,i,λ,j,x) + ∂∂(uⱼ,j,μ,i,x) + δ(i,j)*Δ(uⱼ,μ,x)
        end
    end |> SVector
end

elastic_ad(u, λ, μ) = x->elastic_ad(u, λ, μ, x)

elastic_ad(u, x) = elastic_ad(u, x->1, x->1, x)
elastic_ad(u) = x->elastic_ad(u,x)


function stress_ad(u, λ, μ, n, x)
    n = length(x)

    map(ntuple(k->(mod1(k,n), fld1(k,n)), n*m)) do (i,j)
        uⱼ = e(u,j)

        # nᵢλ∂ⱼuⱼ + nⱼμ∂ᵢuⱼ + δᵢⱼnₖμ∂ₖuⱼ
        # δᵢₗnₗλ∂ⱼuⱼ + δⱼₗnₗμ∂ᵢuⱼ + δᵢⱼδₖₗnₗμ∂ₖuⱼ
        # (δᵢₗλ∂ⱼuⱼ + δⱼₗμ∂ᵢuⱼ + δᵢⱼδₖₗμ∂ₖuⱼ)nₗ
        # (δᵢⱼλ∂ₖuₖ + μ∂ᵢuⱼ + μ∂ⱼuᵢ)nⱼ

        λ(x)*∂(uⱼ, j) + ...
    end |> SMatrix{n,n}
end
