using Test

using Diffinitive.SbpOperators
using Diffinitive.Grids
# using Diffinitive.LazyTensors
using StaticArrays
using ForwardDiff

using LinearAlgebra

operator_path = sbp_operators_path()*"standard_diagonal.toml"
stencil_set = read_stencil_set(operator_path, order = 4)

function test_accuracy(g; L̄, u, Lu, broken=false, debug=false, kwargs...)
    ū = map(u, g)
    Lū = map(Lu, g)
    L̄ū = L̄*ū

    if debug
        @show norm(L̄ū-Lū)/norm(Lū), norm(L̄ū-Lū), norm(Lū)
    end

    @test isapprox(L̄ū, Lū; kwargs...) broken=broken
end

function test_accuracy(g_domain, g_range; L̄, u, Lu, broken=false, debug=false, kwargs...)
    ū = map(u, g_domain)
    Lū = map(Lu, g_range)

    L̄ū = L̄*ū

    if debug
        @show norm(L̄ū-Lū)/norm(Lū), norm(L̄ū-Lū), norm(Lū)
    end

    @test isapprox(L̄ū, Lū; kwargs...) broken=broken
end

## Automatic differentiation
onehot(k,N) = SVector(ntuple(i->k==i ? 1 : 0,N))
index_tuple(x) = tuple_range(length(x))

δ(i,j) = i==j ? 1 : 0

e(u, i, x) = u(x)[i]
e(u,i) = x->e(u,i,x)

∂(f,d::AbstractArray,x) = ForwardDiff.derivative(s->f(x+s*d),0)
∂(f,d::AbstractArray) = x->∂(f,d,x)

∂(f,i::Int,x) = ∂(f,onehot(i, length(x)),x)
∂(f,i::Int) = x->∂(f,i,x)

∂∂(f, i, j, x::AbstractArray) = ∂(∂(f,j),i,x)
∂∂(f, i, j) = x->∂∂(f, i, j, x)

∂∂(f, i, σ, j, x::AbstractArray) = ∂(x->σ(x)*∂(f,j,x),i,x)
∂∂(f, i, σ, j) = x->∂∂(f, i, σ, j, x)

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

    _smatrix(n,n) do i,j
        uᵢ = e(u,i)
        uⱼ = e(u,j)

        # σᵢⱼ = δᵢⱼλ∂ₖuₖ + μ∂ᵢuⱼ + μ∂ⱼuᵢ
        δ(i,j)*λ(x)*div(u,x) + μ(x)*(∂(uⱼ,i,x) + ∂(uᵢ,j,x))
    end
end


stress_ad(u, λ, μ) = x->stress_ad(u, λ, μ, x)

stress_ad(u, x) = stress_ad(u, x->1, x->1, x)
stress_ad(u) = x->stress_ad(u,x)


function traction_ad(u,λ,μ,c,boundary,ξ)
    x = c(ξ)
    σ = stress_ad(u,λ,μ,x)
    n̂ = boundary_normal(c,boundary,ξ)

    return σ*n̂
end

traction_ad(u,λ,μ,c,boundary) = ξ->traction_ad(u,λ,μ,c,boundary,ξ)


function normal_traction_ad(u,λ,μ,c,boundary,ξ)
    x = c(ξ)
    σ = stress_ad(u,λ,μ,x)
    n̂ = boundary_normal(c,boundary,ξ)

    return dot(n̂,σ,n̂)
end

normal_traction_ad(u,λ,μ,c,boundary) = ξ->normal_traction_ad(u,λ,μ,c,boundary,ξ)


function tangential_traction_ad(u,λ,μ,c,boundary,ξ)
    x = c(ξ)
    n̂ = boundary_normal(c,boundary,ξ)
    tₙ = normal_traction_ad(u,λ,μ,c,boundary,ξ)
    T = traction_ad(u,λ,μ,c,boundary,ξ)

    return T - tₙ*n̂
end

tangential_traction_ad(u,λ,μ,c,boundary) = ξ->tangential_traction_ad(u,λ,μ,c,boundary,ξ)


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


const c_2d = with_jacobian(unitsquare(), ForwardDiff.jacobian) do (ξ,η)
    @SVector[1.2ξ+0.2η, 0.9η+ξ/2]
end

const c_3d = with_jacobian(unitcube(), ForwardDiff.jacobian) do (ξ,η,γ)
    @SVector[
        1.2ξ + 0.2η + 0.3γ,
        0.5ξ + 1.1η + 0.4γ,
        0.2ξ + 0.1η + 0.9γ,
    ]
end


## Analytic solutions
∥(u,v) = u⋅unit(v)*unit(v)
⟂(u,v) = u-(u∥v)
unit(v) = v/norm(v)


function plane_wave(kₚ,kₛ,k̂,u₀,x)
    k̄ₚ = kₚ*k̂
    k̄ₛ = kₛ*k̂

    return (u₀∥k̂)*cis(k̄ₚ⋅x) + (u₀⟂k̂)*cis(k̄ₛ⋅x)
end



function elastic_greens_function(μ,kₚ,kₛ,x)
    r(x) = norm(x)

    f(x) = g(kₛ,r(x)) - g(kₚ, r(x))

    return _smatrix(3,3) do i, j
        1/μ*δ(i,j)*g(kₛ,r(x)) + ∂∂(f,i,j,x)
    end
end


g(k,r) = cis(k*r)/(4π*r)


# dimension_cases = ["2D", "3D"]
dimension_cases = ["2D"]

function_cases = Dict(
    "2D" => [
        "u = [x, y²], λ = 1, μ = 1" => (;
            u = x -> @SVector[x[1],x[2]^2],
            λ = x -> 1.,
            μ = x ->1.,
        ),
        "u = [x, y²], λ = y, μ = x" => (;
            u = x -> @SVector[x[1],x[2]^2],
            λ = x -> x[2],
            μ = x -> x[1],
        ),
        "u = [x, y²], λ = x², μ = y" => (;
            u = x -> @SVector[x[1],x[2]^2],
            λ = x -> x[1]^2,
            μ = x -> x[2],
        ),

        "u = [y, x], λ = x, μ = y" => (;
            u = x -> @SVector[x[2],x[1]],
            λ = x -> x[1],
            μ = x -> x[2],
        ),

        "u = [y, x], λ = y, μ = xy" => (;
            u = x -> @SVector[x[2],x[1]],
            λ = x -> x[2],
            μ = x -> x[1]*x[2],
        ),
    ],
    "3D" => [
        "u = [x, y², xz], λ = 1, μ = 1" => (;
            u = x -> @SVector[x[1], x[2]^2, x[1]*x[3]],
            λ = x -> 1.,
            μ = x -> 1.,
        ),
    ],
)

grid_cases = Dict(
    "2D" => [
        "EquidistantGrid" => (ps = unitsquare(Float64), sz = (41, 41),     c = with_jacobian(identity, unitsquare(Float64), ForwardDiff.jacobian)),
        "MappedGrid"      => (ps = c_2d,                sz = (41, 41),     c = c_2d),
    ],
    "3D" => [
        "EquidistantGrid" => (ps = unitcube(Float64),   sz = (21, 21, 21), c = with_jacobian(identity, unitcube(Float64), ForwardDiff.jacobian)),
        "MappedGrid"      => (ps = c_3d,                sz = (21, 21, 21), c = c_3d),
    ],
)

 material_cases = [
    "λ = 1, μ = 1" => (;
        λ = x -> 1.,
        μ = x -> 1.,
    ),
    "λ = 1+0.3sin(||x||), μ = 1+0.2cos(||x||)" => (;
        λ = x -> 1. + 0.3sin(norm(x)),
        μ = x -> 1. + 0.2cos(norm(x)),
    ),
]


@testset "elastic_isotropic" begin
    test_params = Dict(
        "2D" => Dict(
            "EquidistantGrid" => Dict(
                "u = [x, y²], λ = 1, μ = 1" => (;rtol = 1e-12),
                "u = [x, y²], λ = y, μ = x" => (;rtol = 1e-12),
                "u = [x, y²], λ = x², μ = y" => (;rtol = 1e-12),
                "u = [y, x], λ = x, μ = y" => (;rtol = 1e-12),
                "u = [y, x], λ = y, μ = xy" => (;rtol = 1e-12),
            ),
            "MappedGrid" => Dict(
                "u = [x, y²], λ = 1, μ = 1" => (;rtol = 1e-12),
                "u = [x, y²], λ = y, μ = x" => (;rtol = 1e-12),
                "u = [x, y²], λ = x², μ = y" => (;rtol = 1e-4),
                "u = [y, x], λ = x, μ = y" => (;rtol = 1e-11),
                "u = [y, x], λ = y, μ = xy" => (;rtol = 1e-11),
            ),
        ),
        "3D" => Dict(
            "EquidistantGrid" => Dict(
                "u = [x, y², xz], λ = 1, μ = 1" => (;rtol = 1e-13),
            ),
            "MappedGrid" => Dict(
                "u = [x, y², xz], λ = 1, μ = 1" => (;rtol = 1e-12),
            ),
        ),
    )
    @testset "$dim" for dim ∈ dimension_cases
        @testset "$grid_name" for (grid_name, (;ps, sz)) ∈ grid_cases[dim]
            g = equidistant_grid(ps, sz...)
            @testset "$case_name" for (case_name, (;u, λ, μ)) ∈ function_cases[dim]
                λ̄ = map(λ, g)
                μ̄ = map(μ, g)
                test_accuracy(g;
                    L̄ = elastic_isotropic(g, λ̄, μ̄, stencil_set),
                    u = u,
                    Lu = elastic_ad(u,λ,μ),
                    test_params[dim][grid_name][case_name]...,
                )
            end
        end
    end
end


@testset "traction_isotropic" begin
    test_params = Dict(
        "2D" => Dict(
            "EquidistantGrid" => Dict(
                "u = [x, y²], λ = 1, μ = 1" => (;rtol=1e-14),
                "u = [x, y²], λ = y, μ = x" => (;atol=1e-12),
                "u = [x, y²], λ = x², μ = y" => (;rtol=1e-14),
                "u = [y, x], λ = x, μ = y" => (;atol=1e-13),
                "u = [y, x], λ = y, μ = xy" => (;atol=1e-13),
            ),
            "MappedGrid" => Dict(
                "u = [x, y²], λ = 1, μ = 1" => (;rtol=1e-12),
                "u = [x, y²], λ = y, μ = x" => (;rtol=1e-12),
                "u = [x, y²], λ = x², μ = y" => (;rtol=1e-12),
                "u = [y, x], λ = x, μ = y" => (;rtol=1e-12),
                "u = [y, x], λ = y, μ = xy" => (;rtol=1e-12),
            ),
        ),
        "3D" => Dict(
            "EquidistantGrid" => Dict(
                "u = [x, y², xz], λ = 1, μ = 1" => (;rtol=1e12),
            ),
            "MappedGrid" => Dict(
                "u = [x, y², xz], λ = 1, μ = 1" => (;rtol=1e12),
            ),
        ),
    )

    @testset "$dim" for dim ∈ dimension_cases
        @testset "$grid_name" for (grid_name, (;ps, sz, c)) ∈ grid_cases[dim]
            g = equidistant_grid(ps, sz...)
            @testset "$case_name" for (case_name, (;u, λ, μ)) ∈ function_cases[dim]
                @testset "$bid" for bid ∈ boundary_identifiers(g)
                    λ̄ = map(λ, g)
                    μ̄ = map(μ, g)
                    gᵧ = boundary_grid(g, bid)

                    if gᵧ isa MappedGrid
                        gᵧ = logical_grid(gᵧ)
                    end
                    test_accuracy(g, gᵧ;
                        L̄ = traction_isotropic(g, λ̄, μ̄, stencil_set, bid),
                        u = u,
                        Lu = traction_ad(u, λ, μ, c, bid),
                        test_params[dim][grid_name][case_name]...,
                    )
                end
            end
        end
    end
end


@testset "normal_traction_isotropic" begin
    test_params = Dict(
        "2D" => Dict(
            "EquidistantGrid" => Dict(
                "u = [x, y²], λ = 1, μ = 1" => (;rtol=1e-12),
                "u = [x, y²], λ = y, μ = x" => (;atol=1e-12),
                "u = [x, y²], λ = x², μ = y" => (;rtol=1e-12),
                "u = [y, x], λ = x, μ = y" => (;atol=1e-12),
                "u = [y, x], λ = y, μ = xy" => (;atol=1e-12),
            ),
            "MappedGrid" => Dict(
                "u = [x, y²], λ = 1, μ = 1" => (;rtol=1e-12),
                "u = [x, y²], λ = y, μ = x" => (;rtol=1e-12),
                "u = [x, y²], λ = x², μ = y" => (;rtol=1e-12),
                "u = [y, x], λ = x, μ = y" => (;rtol=1e-12),
                "u = [y, x], λ = y, μ = xy" => (;rtol=1e-12),
            ),
        ),
        "3D" => Dict(
            "EquidistantGrid" => Dict(
                "u = [x, y², xz], λ = 1, μ = 1" => (;rtol=1e-12),
            ),
            "MappedGrid" => Dict(
                "u = [x, y², xz], λ = 1, μ = 1" => (;rtol=1e-12),
            ),
        ),
    )

    @testset "$dim" for dim ∈ dimension_cases
        @testset "$grid_name" for (grid_name, (;ps, sz, c)) ∈ grid_cases[dim]
            g = equidistant_grid(ps, sz...)
            @testset "$case_name" for (case_name, (;u, λ, μ)) ∈ function_cases[dim]
                @testset "$bid" for bid ∈ boundary_identifiers(g)
                    λ̄ = map(λ, g)
                    μ̄ = map(μ, g)
                    gᵧ = boundary_grid(g, bid)

                    if gᵧ isa MappedGrid
                        gᵧ = logical_grid(gᵧ)
                    end
                    test_accuracy(g, gᵧ;
                        L̄ = normal_traction_isotropic(g, λ̄, μ̄, stencil_set, bid),
                        u = u,
                        Lu = normal_traction_ad(u, λ, μ, c, bid),
                        test_params[dim][grid_name][case_name]...,
                    )
                end
            end
        end
    end
end


@testset "tangential_traction_isotropic" begin
    test_params = Dict(
        "2D" => Dict(
            "EquidistantGrid" => Dict(
                "u = [x, y²], λ = 1, μ = 1" => (;atol=1e-12),
                "u = [x, y²], λ = y, μ = x" => (;atol=1e-12),
                "u = [x, y²], λ = x², μ = y" => (;atol=1e-12),
                "u = [y, x], λ = x, μ = y" => (;rtol=1e-12),
                "u = [y, x], λ = y, μ = xy" => (;rtol=1e-12),
            ),
            "MappedGrid" => Dict(
                "u = [x, y²], λ = 1, μ = 1" => (;rtol=1e-12),
                "u = [x, y²], λ = y, μ = x" => (;rtol=1e-12),
                "u = [x, y²], λ = x², μ = y" => (;rtol=1e-12),
                "u = [y, x], λ = x, μ = y" => (;rtol=1e-12),
                "u = [y, x], λ = y, μ = xy" => (;rtol=1e-12),
            ),
        ),
        "3D" => Dict(
            "EquidistantGrid" => Dict(
                "u = [x, y², xz], λ = 1, μ = 1" => (;atol=1e-12),
            ),
            "MappedGrid" => Dict(
                "u = [x, y², xz], λ = 1, μ = 1" => (;rtol=1e-12),
            ),
        ),
    )

    @testset "$dim" for dim ∈ dimension_cases
        @testset "$grid_name" for (grid_name, (;ps, sz, c)) ∈ grid_cases[dim]
            g = equidistant_grid(ps, sz...)
            @testset "$case_name" for (case_name, (;u, λ, μ)) ∈ function_cases[dim]
                @testset "$bid" for bid ∈ boundary_identifiers(g)
                    λ̄ = map(λ, g)
                    μ̄ = map(μ, g)
                    gᵧ = boundary_grid(g, bid)

                    if gᵧ isa MappedGrid
                        gᵧ = logical_grid(gᵧ)
                    end
                    test_accuracy(g, gᵧ;
                        L̄ = tangential_traction_isotropic(g, λ̄, μ̄, stencil_set, bid),
                        u = u,
                        Lu = tangential_traction_ad(u, λ, μ, c, bid),
                        test_params[dim][grid_name][case_name]...,
                    )
                end
            end
        end
    end
end


@testset "T = tₜ+tₙn̂" begin
    @testset "$dim" for dim ∈ dimension_cases
        @testset "$grid_name" for (grid_name, (;ps, sz)) ∈ grid_cases[dim]
            g = equidistant_grid(ps, sz...)
            @testset "$case_name" for (case_name, (;λ, μ)) ∈ material_cases
                λ̄ = map(λ, g)
                μ̄ = map(μ, g)

                u = rand(SVector{ndims(g)}, size(g))

                @testset "$boundary" for boundary ∈ boundary_identifiers(g)
                    T = traction_isotropic(g, λ̄, μ̄, stencil_set, boundary)
                    tₜ = tangential_traction_isotropic(g, λ̄, μ̄, stencil_set, boundary)
                    tₙ = normal_traction_isotropic(g, λ̄, μ̄, stencil_set, boundary)
                    n̂ = boundary_normal(g, boundary)

                    @test T*u ≈ tₜ*u + (tₙ*u).*n̂
                end
            end
        end
    end
end


@testset "SBP-properties" begin
    ip(u,H,v) = mapreduce(⋅, +, u , H*v)

    @testset "$dim" for dim ∈ dimension_cases
        @testset "$grid_name" for (grid_name, (;ps, sz)) ∈ grid_cases[dim]
            g = equidistant_grid(ps, sz...)
            @testset "$case_name" for (case_name, parameters) ∈ material_cases
                (;λ, μ) = parameters
                λ̄ = map(λ, g)
                μ̄ = map(μ, g)

                # Test that the summation by parts property for the elastic operator
                # ( vᵢ, [Eu]ᵢ)_Ω - ([Ev]ᵢ, uᵢ)_Ω = (vᵢ, [Tu]ᵢ )_∂Ω - ([Tv]ᵢ, uᵢ)_∂Ω
                # Holds

                E = elastic_isotropic(g, λ̄, μ̄, stencil_set)

                u = rand(SVector{ndims(g)}, size(g))
                v = rand(SVector{ndims(g)}, size(g))

                H = inner_product(g, stencil_set)

                volume_term = ip(v,H,E*u) - ip(E*v,H,u)
                boundary_term = sum(boundary_identifiers(g)) do boundary
                    e = boundary_restriction(g, stencil_set, boundary)
                    T = traction_isotropic(g, λ̄, μ̄, stencil_set, boundary)
                    Hᵧ = inner_product(boundary_grid(g, boundary), stencil_set)

                    ip(e*v, Hᵧ, T*u) - ip(T*v, Hᵧ, e*u)
                end

                @test volume_term ≈ boundary_term
           end
        end
    end
end
