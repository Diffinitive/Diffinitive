using Test

using Diffinitive.SbpOperators
using Diffinitive.Grids
# using Diffinitive.LazyTensors
using StaticArrays
using ForwardDiff


test_grid(::Type{<:TensorGrid}) = equidistant_grid(unitsquare(Float64),41,41)

const c = with_jacobian(unitsquare(), ForwardDiff.jacobian) do (ξ,η)
    @SVector[1.2ξ+0.2η, 0.9η+ξ/2]
end

test_grid(::Type{<:MappedGrid}) = equidistant_grid(c, n, m)


@testset "elastic_isotropic" begin
    @testset "EquidistantGrid" begin
        @testset "2D" begin
            @test_broken false
        end

        @testset "3D" begin
            @test_broken false
        end
    end

    @testset "MappedGrid" begin
        @testset "2D" begin
            @test_broken false
        end

        @testset "3D" begin
            @test_broken false
        end
    end
end

@testset "traction_isotropic" begin
    @testset "EquidistantGrid" begin
        @testset "2D" begin
            @test_broken false
        end

        @testset "3D" begin
            @test_broken false
        end
    end

    @testset "MappedGrid" begin
        @testset "2D" begin
            @test_broken false
        end

        @testset "3D" begin
            @test_broken false
        end
    end
end

@testset "normal_traction_isotropic" begin
    @testset "EquidistantGrid" begin
        @testset "2D" begin
            @test_broken false
        end

        @testset "3D" begin
            @test_broken false
        end
    end

    @testset "MappedGrid" begin
        @testset "2D" begin
            @test_broken false
        end

        @testset "3D" begin
            @test_broken false
        end
    end
end

@testset "tangential_traction_isotropic" begin
    @testset "EquidistantGrid" begin
        @testset "2D" begin
            @test_broken false
        end

        @testset "3D" begin
            @test_broken false
        end
    end

    @testset "MappedGrid" begin
        @testset "2D" begin
            @test_broken false
        end

        @testset "3D" begin
            @test_broken false
        end
    end
end


@testset "SBP-properties" begin
    # TODO: test for a few random vectors
    @testset "EquidistantGrid" begin
        @testset "2D" begin
            @test_broken false
        end

        @testset "3D" begin
            @test_broken false
        end
    end

    @testset "MappedGrid" begin
        @testset "2D" begin
            @test_broken false
        end

        @testset "3D" begin
            @test_broken false
        end
    end
end

# SBP-factorization
# Accuracy
    # Exact polynomials
    # something more complicated

# 2D
# 3D


## Automatic differentiation
onehot(k,N) = SVector(ntuple(i->k==i ? 1 : 0,N))
tuple_range(n) = ntuple(identity, n)
index_tuple(x) = tuple_range(length(x))

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

Δ(f,σ,x) = sum(k->∂∂(f,k,σ,k,x), index_tuple(x))
Δ(f,σ) = x->Δ(f,σ,x)

div(f, x) = sum(k->∂(e(f,k),k,x), index_tuple(x))
div(f) = x->div(f,x)

grad(f, x) = map(i->∂(f,i,x), index_tuple(x))
grad(f) = x->grad(f,x)

# function J(f, x)
#     n = length(f(zero(x)))
#     m = length(x)

#     _smatrix(n,m) do i,j
#         @inline
#         ∂(e(f,i),j,x)
#     end
# end
# Can the above be made type-stable?


function elastic_ad(u, λ, μ, x)
    map(index_tuple(x)) do i
        sum(index_tuple(x)) do j
            uⱼ = e(u,j)
            # ∂ᵢλ∂ⱼuⱼ + ∂ⱼμ∂ᵢuⱼ + δᵢⱼ∂ₖμ∂ₖuⱼ
            ∂∂(uⱼ,i,λ,j,x) + ∂∂(uⱼ,j,μ,i,x) + δ(i,j)*Δ(uⱼ,μ,x)
        end
    end |> SVector
end

elastic_ad(u, λ, μ) = x->elastic_ad(u, λ, μ, x)

elastic_ad(u, x) = elastic_ad(u, x->1, x->1, x)
elastic_ad(u) = x->elastic_ad(u,x)


function stress_ad(u, λ, μ, x)
    n = length(x)

    _smatrix(n,n) do (i,j)
        uᵢ = e(u,i)
        uⱼ = e(u,j)

        # σᵢⱼ = δᵢⱼλ∂ₖuₖ + μ∂ᵢuⱼ + μ∂ⱼuᵢ
        δ(i,j)*λ(x)*div(u,x) + μ(x)*(∂(uⱼ,i,x) + ∂(uᵢ,j,x))
    end
end


stress_ad(u, λ, μ) = x->stress_ad(u, λ, μ, x)

stress_ad(u, x) = stress_ad(u, x->1, x->1, x)
stress_ad(u) = x->stress_ad(u,x)


## Helpers
function _smatrix(f,n,m)
    map(ntuple(k->(mod1(k,n), fld1(k,n)), n*m)) do (i,j)
        @inline
        f(i,j)
    end |> SMatrix{n,m}
end

function _smatrix(tt::NTuple{N, NTuple{M, Any}}) where {N,M}
    return _smatrix((i,j)->tt[i][j],N,M)
end
