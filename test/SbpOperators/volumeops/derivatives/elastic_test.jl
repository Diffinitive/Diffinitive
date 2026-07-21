using Test

using Diffinitive.SbpOperators
using Diffinitive.Grids

using StaticArrays
using AutomaticCalculus: e, ∂∂, δ, Δ, J, index_tuple

using LinearAlgebra

const operator_path = sbp_operators_path()*"standard_diagonal.toml"
const stencil_set = read_stencil_set(operator_path, order = 4)

## Automatic differentiation
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


## Helpers
function test_accuracy(g; L̄, u, Lu, broken=false, debug=false, kwargs...)
    ū = map(u, g)
    Lū = map(Lu, g)
    L̄ū = L̄*ū

    if debug
        @show norm(L̄ū-Lū)/norm(Lū), norm(L̄ū-Lū), norm(Lū)
    end

    @test isapprox(L̄ū, Lū; kwargs...) broken=broken
end

function _smatrix(f,n,m)
    map(ntuple(k->(mod1(k,n), fld1(k,n)), n*m)) do (i,j)
        @inline
        f(i,j)
    end |> SMatrix{n,m}
end

function _smatrix(tt::NTuple{N, NTuple{M, Any}}) where {N,M}
    return _smatrix((i,j)->tt[i][j],N,M)
end


const c_2d = with_jacobian(unitsquare(), J) do (ξ,η)
    @SVector[1.2ξ+0.2η, 0.9η+ξ/2]
end

const c_3d = with_jacobian(unitcube(), J) do (ξ,η,γ)
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

## Test sets
# dimension_cases = ["2D", "3D"]  # TODO: "Runtime for 3D tests are very long. Investigate
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
        "EquidistantGrid" => (ps = unitsquare(Float64), sz = (41, 41),     c = with_jacobian(identity, unitsquare(Float64), jacobian)),
        "MappedGrid"      => (ps = c_2d,                sz = (41, 41),     c = c_2d),
    ],
    "3D" => [
        "EquidistantGrid" => (ps = unitcube(Float64),   sz = (21, 21, 21), c = with_jacobian(identity, unitcube(Float64), jacobian)),
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


@testset "elastic" begin
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
                    L̄ = elastic(g, λ̄, μ̄, stencil_set),
                    u = u,
                    Lu = elastic_ad(u,λ,μ),
                    test_params[dim][grid_name][case_name]...,
                )
            end
        end
    end
end


@testset "SBP-properties" begin
    # Test that the summation by parts property for the elastic operator
    # ( vᵢ, [Eu]ᵢ)_Ω - ([Ev]ᵢ, uᵢ)_Ω = (vᵢ, [Tu]ᵢ )_∂Ω - ([Tv]ᵢ, uᵢ)_∂Ω
    # holds
    ip(u,H,v) = mapreduce(⋅, +, u , H*v)

    @testset "$dim" for dim ∈ dimension_cases
        @testset "$grid_name" for (grid_name, (;ps, sz)) ∈ grid_cases[dim]
            g = equidistant_grid(ps, sz...)
            @testset "$case_name" for (case_name, parameters) ∈ material_cases
                (;λ, μ) = parameters
                λ̄ = map(λ, g)
                μ̄ = map(μ, g)
        
                E = elastic(g, λ̄, μ̄, stencil_set)

                u = rand(SVector{ndims(g)}, size(g))
                v = rand(SVector{ndims(g)}, size(g))

                H = inner_product(g, stencil_set)

                volume_term = ip(v,H,E*u) - ip(E*v,H,u)
                boundary_term = sum(boundary_identifiers(g)) do boundary
                    e = boundary_restriction(g, stencil_set, boundary)
                    T = traction(g, λ̄, μ̄, stencil_set, boundary)
                    Hᵧ = inner_product(boundary_grid(g, boundary), stencil_set)

                    ip(e*v, Hᵧ, T*u) - ip(T*v, Hᵧ, e*u)
                end

                @test volume_term ≈ boundary_term
           end
        end
    end
end
