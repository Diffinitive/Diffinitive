using Test

using Diffinitive.SbpOperators
using Diffinitive.Grids
# using Diffinitive.LazyTensors
using StaticArrays
using ForwardDiff

using LinearAlgebra

const operator_path = sbp_operators_path()*"standard_diagonal.toml"
const stencil_set = read_stencil_set(operator_path, order = 4)

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
    n̂ = normal(c,boundary,ξ)

    return σ*n̂
end

traction_ad(u,λ,μ,c,boundary) = ξ->traction_ad(u,λ,μ,c,boundary,ξ)


function normal_traction_ad(u,λ,μ,c,boundary,ξ)
    x = c(ξ)
    σ = stress_ad(u,λ,μ,x)
    n̂ = normal(c,boundary,ξ)

    return dot(n̂,σ,n̂)
end

normal_traction_ad(u,λ,μ,c,boundary) = ξ->normal_traction_ad(u,λ,μ,c,boundary,ξ)


function tangential_traction_ad(u,λ,μ,c,boundary,ξ)
    x = c(ξ)
    n̂ = normal(c,boundary,ξ)
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


function_cases_2d = [
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
]

function_cases_3d = [
    "u = [x, y², xz], λ = 1, μ = 1" => (;
        u = x -> @SVector[x[1], x[2]^2, x[1]*x[3]],
        λ = x -> 1.,
        μ = x -> 1.,
    ),
]


@testset "elastic_isotropic" begin
    @testset "2D" begin
        grid_cases = [
            "EquidistantGrid" => equidistant_grid(unitsquare(Float64), 41, 41),
            "MappedGrid" => equidistant_grid(c_2d, 41, 41),
        ]

        rtols = Dict(
            "EquidistantGrid" => Dict(
                "u = [x, y²], λ = 1, μ = 1" => 1e-12,
                "u = [x, y²], λ = y, μ = x" => 1e-12,
                "u = [x, y²], λ = x², μ = y" => 1e-12,
                "u = [y, x], λ = x, μ = y" => 1e-12,
                "u = [y, x], λ = y, μ = xy" => 1e-12,
            ),
            "MappedGrid" => Dict(
                "u = [x, y²], λ = 1, μ = 1" => 1e-12,
                "u = [x, y²], λ = y, μ = x" => 1e-12,
                "u = [x, y²], λ = x², μ = y" => 1e-4,
                "u = [y, x], λ = x, μ = y" => 1e-11,
                "u = [y, x], λ = y, μ = xy" => 1e-11,
            ),
        )

        @testset "$grid_name" for (grid_name, g) ∈ grid_cases
            @testset "$case_name" for (case_name, parameters) ∈ function_cases_2d
                (;u, λ, μ) = parameters
                λ̄ = map(λ, g)
                μ̄ = map(μ, g)
                test_accuracy(g;
                    L̄ = elastic_isotropic(g, λ̄, μ̄, stencil_set),
                    u = u,
                    Lu = elastic_ad(u,λ,μ),
                    rtol = rtols[grid_name][case_name],
                )
            end
        end
    end


    @testset "3D" begin
        grid_cases = [
            "EquidistantGrid" => equidistant_grid(unitcube(Float64), 21, 21, 21),
            "MappedGrid" => equidistant_grid(c_3d, 21, 21, 21),
        ]


        rtols = Dict(
            "EquidistantGrid" => Dict(
                "u = [x, y², xz], λ = 1, μ = 1" => 1e-13,
            ),
            "MappedGrid" => Dict(
                "u = [x, y², xz], λ = 1, μ = 1" => 1e-12,
            ),
        )

        @testset "$grid_name" for (grid_name, g) ∈ grid_cases
            @testset "$case_name" for (case_name, parameters) ∈ function_cases_3d
                (;u, λ, μ) = parameters
                λ̄ = map(λ, g)
                μ̄ = map(μ, g)
                test_accuracy(g;
                    L̄ = elastic_isotropic(g, λ̄, μ̄, stencil_set),
                    u = u,
                    Lu = elastic_ad(u,λ,μ),
                    rtol = rtols[grid_name][case_name],
                )
            end
        end
    end
end


@testset "traction_isotropic" begin
    @testset "2D" begin
        @testset "EquidistantGrid" begin
            test_params = Dict(
                "u = [x, y²], λ = 1, μ = 1" => (;rtol=1e-14),
                "u = [x, y²], λ = y, μ = x" => (;atol=1e-12),
                "u = [x, y²], λ = x², μ = y" => (;rtol=1e-14),
                "u = [y, x], λ = x, μ = y" => (;atol=1e-13),
                "u = [y, x], λ = y, μ = xy" => (;atol=1e-13),
            )
            s = unitsquare(Float64)
            g = equidistant_grid(s, 41, 41)
            c = with_jacobian(identity, s, ForwardDiff.jacobian) # Needed for the AD
            @testset "$case_name" for (case_name, parameters) ∈ function_cases_2d
                @testset "$bid" for bid ∈ boundary_identifiers(g)
                    (;u, λ, μ) = parameters
                    λ̄ = map(λ, g)
                    μ̄ = map(μ, g)
                    g̃ᵧ = boundary_grid(g, bid)
                    test_accuracy(g, g̃ᵧ;
                        L̄ = traction_isotropic(g, λ̄, μ̄, stencil_set, bid),
                        u = u,
                        Lu = traction_ad(u, λ, μ, c, bid),
                        test_params[case_name]...,
                    )
                end
            end
        end

        @testset "MappedGrid" begin
            test_params = Dict(
                "u = [x, y²], λ = 1, μ = 1" => (;rtol=1e-12),
                "u = [x, y²], λ = y, μ = x" => (;rtol=1e-12),
                "u = [x, y²], λ = x², μ = y" => (;rtol=1e-12),
                "u = [y, x], λ = x, μ = y" => (;rtol=1e-12),
                "u = [y, x], λ = y, μ = xy" => (;rtol=1e-12),
            )
            g = equidistant_grid(c_2d, 41, 41)
            @testset "$case_name" for (case_name, parameters) ∈ function_cases_2d
                @testset "$bid" for bid ∈ boundary_identifiers(g)
                    (;u, λ, μ) = parameters
                    λ̄ = map(λ, g)
                    μ̄ = map(μ, g)
                    g̃ᵧ = boundary_grid(g, bid) |> logical_grid
                    test_accuracy(g, g̃ᵧ;
                        L̄ = traction_isotropic(g, λ̄, μ̄, stencil_set, bid),
                        u = u,
                        Lu = traction_ad(u,λ,μ,c_2d,bid),
                        test_params[case_name]...,
                    )
                end
            end
        end
    end

    @testset "3D" begin
        @testset "EquidistantGrid" begin
            test_params = Dict(
                "u = [x, y², xz], λ = 1, μ = 1" => (;rtol=1e12),
            )
            s = unitcube(Float64)
            g = equidistant_grid(s, 21, 21, 21)
            c = with_jacobian(identity, s, ForwardDiff.jacobian) # Needed for the AD
            @testset "$case_name" for (case_name, parameters) ∈ function_cases_3d
                @testset "$bid" for bid ∈ boundary_identifiers(g)
                    (;u, λ, μ) = parameters
                    λ̄ = map(λ, g)
                    μ̄ = map(μ, g)
                    g̃ᵧ = boundary_grid(g, bid)
                    test_accuracy(g, g̃ᵧ;
                        L̄ = traction_isotropic(g, λ̄, μ̄, stencil_set, bid),
                        u = u,
                        Lu = traction_ad(u, λ, μ, c, bid),
                        test_params[case_name]...,
                    )
                end
            end
        end

        @testset "MappedGrid" begin
            test_params = Dict(
                "u = [x, y², xz], λ = 1, μ = 1" => (;rtol=1e12),
            )
            g = equidistant_grid(c_3d, 21, 21, 21)
            @testset "$case_name" for (case_name, parameters) ∈ function_cases_3d
                @testset "$bid" for bid ∈ boundary_identifiers(g)
                    (;u, λ, μ) = parameters
                    λ̄ = map(λ, g)
                    μ̄ = map(μ, g)
                    g̃ᵧ = boundary_grid(g, bid) |> logical_grid
                    test_accuracy(g, g̃ᵧ;
                        L̄ = traction_isotropic(g, λ̄, μ̄, stencil_set, bid),
                        u = u,
                        Lu = traction_ad(u,λ,μ,c_3d,bid),
                        test_params[case_name]...,
                    )
                end
            end
        end
    end
end


@testset "normal_traction_isotropic" begin
    @testset "2D" begin
        @testset "EquidistantGrid" begin
            test_params = Dict(
                "u = [x, y²], λ = 1, μ = 1" => (;rtol=1e-12),
                "u = [x, y²], λ = y, μ = x" => (;atol=1e-12),
                "u = [x, y²], λ = x², μ = y" => (;rtol=1e-12),
                "u = [y, x], λ = x, μ = y" => (;atol=1e-12),
                "u = [y, x], λ = y, μ = xy" => (;atol=1e-12),
            )
            s = unitsquare(Float64)
            g = equidistant_grid(s, 41, 41)
            c = with_jacobian(identity, s, ForwardDiff.jacobian) # Needed for the AD
            @testset "$case_name" for (case_name, parameters) ∈ function_cases_2d
                @testset "$bid" for bid ∈ boundary_identifiers(g)
                    (;u, λ, μ) = parameters
                    λ̄ = map(λ, g)
                    μ̄ = map(μ, g)
                    g̃ᵧ = boundary_grid(g, bid)
                    test_accuracy(g, g̃ᵧ;
                        L̄ = normal_traction_isotropic(g, λ̄, μ̄, stencil_set, bid),
                        u = u,
                        Lu = normal_traction_ad(u, λ, μ, c, bid),
                        test_params[case_name]...,
                    )
                end
            end
        end

        @testset "MappedGrid" begin
            test_params = Dict(
                "u = [x, y²], λ = 1, μ = 1" => (;rtol=1e-12),
                "u = [x, y²], λ = y, μ = x" => (;rtol=1e-12),
                "u = [x, y²], λ = x², μ = y" => (;rtol=1e-12),
                "u = [y, x], λ = x, μ = y" => (;rtol=1e-12),
                "u = [y, x], λ = y, μ = xy" => (;rtol=1e-12),
            )
            g = equidistant_grid(c_2d, 41, 41)
            @testset "$case_name" for (case_name, parameters) ∈ function_cases_2d
                @testset "$bid" for bid ∈ boundary_identifiers(g)
                    (;u, λ, μ) = parameters
                    λ̄ = map(λ, g)
                    μ̄ = map(μ, g)
                    g̃ᵧ = boundary_grid(g, bid) |> logical_grid
                    test_accuracy(g, g̃ᵧ;
                        L̄ = normal_traction_isotropic(g, λ̄, μ̄, stencil_set, bid),
                        u = u,
                        Lu = normal_traction_ad(u,λ,μ,c_2d,bid),
                        test_params[case_name]...,
                    )
                end
            end
        end
    end

    @testset "3D" begin
        @testset "EquidistantGrid" begin
            test_params = Dict(
                "u = [x, y², xz], λ = 1, μ = 1" => (;rtol=1e-12),
            )
            s = unitcube(Float64)
            g = equidistant_grid(s, 21, 21, 21)
            c = with_jacobian(identity, s, ForwardDiff.jacobian) # Needed for the AD
            @testset "$case_name" for (case_name, parameters) ∈ function_cases_3d
                @testset "$bid" for bid ∈ boundary_identifiers(g)
                    (;u, λ, μ) = parameters
                    λ̄ = map(λ, g)
                    μ̄ = map(μ, g)
                    g̃ᵧ = boundary_grid(g, bid)
                    test_accuracy(g, g̃ᵧ;
                        L̄ = normal_traction_isotropic(g, λ̄, μ̄, stencil_set, bid),
                        u = u,
                        Lu = normal_traction_ad(u, λ, μ, c, bid),
                        test_params[case_name]...,
                    )
                end
            end
        end

        @testset "MappedGrid" begin
            test_params = Dict(
                "u = [x, y², xz], λ = 1, μ = 1" => (;rtol=1e-12),
            )
            g = equidistant_grid(c_3d, 21, 21, 21)
            @testset "$case_name" for (case_name, parameters) ∈ function_cases_3d
                @testset "$bid" for bid ∈ boundary_identifiers(g)
                    (;u, λ, μ) = parameters
                    λ̄ = map(λ, g)
                    μ̄ = map(μ, g)
                    g̃ᵧ = boundary_grid(g, bid) |> logical_grid
                    test_accuracy(g, g̃ᵧ;
                        L̄ = normal_traction_isotropic(g, λ̄, μ̄, stencil_set, bid),
                        u = u,
                        Lu = normal_traction_ad(u,λ,μ,c_3d,bid),
                        test_params[case_name]...,
                    )
                end
            end
        end
    end
end


@testset "tangential_traction_isotropic" begin
    @testset "2D" begin
        @testset "EquidistantGrid" begin
            test_params = Dict(
                "u = [x, y²], λ = 1, μ = 1" => (;atol=1e-12),
                "u = [x, y²], λ = y, μ = x" => (;atol=1e-12),
                "u = [x, y²], λ = x², μ = y" => (;atol=1e-12),
                "u = [y, x], λ = x, μ = y" => (;rtol=1e-12),
                "u = [y, x], λ = y, μ = xy" => (;rtol=1e-12),
            )
            s = unitsquare(Float64)
            g = equidistant_grid(s, 41, 41)
            c = with_jacobian(identity, s, ForwardDiff.jacobian) # Needed for the AD
            @testset "$case_name" for (case_name, parameters) ∈ function_cases_2d
                @testset "$bid" for bid ∈ boundary_identifiers(g)
                    (;u, λ, μ) = parameters
                    λ̄ = map(λ, g)
                    μ̄ = map(μ, g)
                    g̃ᵧ = boundary_grid(g, bid)
                    test_accuracy(g, g̃ᵧ;
                        L̄ = tangential_traction_isotropic(g, λ̄, μ̄, stencil_set, bid),
                        u = u,
                        Lu = tangential_traction_ad(u, λ, μ, c, bid),
                        test_params[case_name]...,
                    )
                end
            end
        end

        @testset "MappedGrid" begin
            test_params = Dict(
                "u = [x, y²], λ = 1, μ = 1" => (;rtol=1e-12),
                "u = [x, y²], λ = y, μ = x" => (;rtol=1e-12),
                "u = [x, y²], λ = x², μ = y" => (;rtol=1e-12),
                "u = [y, x], λ = x, μ = y" => (;rtol=1e-12),
                "u = [y, x], λ = y, μ = xy" => (;rtol=1e-12),
            )
            g = equidistant_grid(c_2d, 41, 41)
            @testset "$case_name" for (case_name, parameters) ∈ function_cases_2d
                @testset "$bid" for bid ∈ boundary_identifiers(g)
                    (;u, λ, μ) = parameters
                    λ̄ = map(λ, g)
                    μ̄ = map(μ, g)
                    g̃ᵧ = boundary_grid(g, bid) |> logical_grid
                    test_accuracy(g, g̃ᵧ;
                        L̄ = tangential_traction_isotropic(g, λ̄, μ̄, stencil_set, bid),
                        u = u,
                        Lu = tangential_traction_ad(u,λ,μ,c_2d,bid),
                        test_params[case_name]...,
                    )
                end
            end
        end
    end

    @testset "3D" begin
        @testset "EquidistantGrid" begin
            test_params = Dict(
                "u = [x, y², xz], λ = 1, μ = 1" => (;atol=1e-12),
            )
            s = unitcube(Float64)
            g = equidistant_grid(s, 21, 21, 21)
            c = with_jacobian(identity, s, ForwardDiff.jacobian) # Needed for the AD
            @testset "$case_name" for (case_name, parameters) ∈ function_cases_3d
                @testset "$bid" for bid ∈ boundary_identifiers(g)
                    (;u, λ, μ) = parameters
                    λ̄ = map(λ, g)
                    μ̄ = map(μ, g)
                    g̃ᵧ = boundary_grid(g, bid)
                    test_accuracy(g, g̃ᵧ;
                        L̄ = tangential_traction_isotropic(g, λ̄, μ̄, stencil_set, bid),
                        u = u,
                        Lu = tangential_traction_ad(u, λ, μ, c, bid),
                        test_params[case_name]...,
                    )
                end
            end
        end

        @testset "MappedGrid" begin
            test_params = Dict(
                "u = [x, y², xz], λ = 1, μ = 1" => (;rtol=1e-12),
            )
            g = equidistant_grid(c_3d, 21, 21, 21)
            @testset "$case_name" for (case_name, parameters) ∈ function_cases_3d
                @testset "$bid" for bid ∈ boundary_identifiers(g)
                    (;u, λ, μ) = parameters
                    λ̄ = map(λ, g)
                    μ̄ = map(μ, g)
                    g̃ᵧ = boundary_grid(g, bid) |> logical_grid
                    test_accuracy(g, g̃ᵧ;
                        L̄ = tangential_traction_isotropic(g, λ̄, μ̄, stencil_set, bid),
                        u = u,
                        Lu = tangential_traction_ad(u,λ,μ,c_3d,bid),
                        test_params[case_name]...,
                    )
                end
            end
        end
    end
end


## TODO: Test that T = tₜ+tₙn̂


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


