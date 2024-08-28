using Test

using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.LazyTensors

import Sbplib.SbpOperators.ConstantInteriorScalingOperator

using StaticArrays
using LinearAlgebra

@testset "Diagonal-stencil inner_product" begin
    Lx = π/2.
    Ly = Float64(π)
    Lz = 1.
    g_1D = equidistant_grid(0.0, Lx, 77)
    g_2D = equidistant_grid((0.0, 0.0), (Lx,Ly), 77, 66)
    g_3D = equidistant_grid((0.0, 0.0, 0.0), (Lx,Ly,Lz), 10, 10, 10)
    @testset "inner_product" begin
        stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        @testset "0D" begin
            H = inner_product(ZeroDimGrid(0.), stencil_set)
            @test H isa LazyTensor{T,0,0} where T
        end
        @testset "1D" begin
            H = inner_product(g_1D, stencil_set)
            @test H isa LazyTensor{T,1,1} where T
        end
        @testset "2D" begin
            H = inner_product(g_2D, stencil_set)
            H_x = inner_product(g_2D.grids[1], stencil_set)
            H_y = inner_product(g_2D.grids[2], stencil_set)
            @test H == H_x⊗H_y
            @test H isa LazyTensor{T,2,2} where T
        end

        # TBD: Should there be more tests?
    end

    @testset "Sizes" begin
        stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        @testset "1D" begin
            H = inner_product(g_1D, stencil_set)
            @test domain_size(H) == size(g_1D)
            @test range_size(H) == size(g_1D)
        end
        @testset "2D" begin
            H = inner_product(g_2D, stencil_set)
            @test domain_size(H) == size(g_2D)
            @test range_size(H) == size(g_2D)
        end
    end

    @testset "Accuracy" begin
        @testset "1D" begin
            v = ()
            for i = 0:4
                f_i(x) = 1/factorial(i)*x^i
                v = (v...,eval_on(g_1D,f_i))
            end
            u = eval_on(g_1D,x->sin(x))

            @testset "2nd order" begin
                stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=2)
                H = inner_product(g_1D, stencil_set)
                for i = 1:2
                    @test sum(H*v[i]) ≈ v[i+1][end] - v[i+1][1] rtol = 1e-14
                end
                @test sum(H*u) ≈ 1. rtol = 1e-4
            end

            @testset "4th order" begin
                stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)
                H = inner_product(g_1D, stencil_set)
                for i = 1:4
                    @test sum(H*v[i]) ≈ v[i+1][end] -  v[i+1][1] rtol = 1e-14
                end
                @test sum(H*u) ≈ 1. rtol = 1e-8
            end
        end

        @testset "2D" begin
            b = 2.1
            v = b*ones(Float64, size(g_2D))
            u = eval_on(g_2D,(x,y)->sin(x)+cos(y))
            @testset "2nd order" begin
                stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=2)
                H = inner_product(g_2D, stencil_set)
                @test sum(H*v) ≈ b*Lx*Ly rtol = 1e-13
                @test sum(H*u) ≈ π rtol = 1e-4
            end
            @testset "4th order" begin
                stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)
                H = inner_product(g_2D, stencil_set)
                @test sum(H*v) ≈ b*Lx*Ly rtol = 1e-13
                @test sum(H*u) ≈ π rtol = 1e-8
            end
        end
    end

    @testset "MappedGrid" begin
        stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        c = Chart(unitsquare()) do (ξ,η)
            @SVector[2ξ + η*(1-η), 3η+(1+η/2)*ξ^2]
        end
        Grids.jacobian(c::typeof(c), (ξ,η)) = @SMatrix[2 1-2η; (2+η)*ξ 3+ξ^2/2]

        mg = equidistant_grid(c, 10,13)

        @test inner_product(mg, stencil_set) isa LazyTensor{<:Any, 2,2}

        @testset "Accuracy" begin
            v = function(x̄)
                log(norm(x̄-@SVector[.5, .5]))/2π + log(norm(x̄-@SVector[1.5, 3]))/2π
            end
            ∇v = function(x̄)
                ∇log(ȳ) = ȳ/(ȳ⋅ȳ)
                ∇log(x̄-@SVector[.5, .5])/2π + ∇log(x̄-@SVector[1.5, 3])/2π
            end

            mg = equidistant_grid(c, 80,80)
            v̄ = map(v, mg)

            @testset for (order, atol) ∈ [(2,1e-3),(4,1e-7)]
                stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=order)

                @test sum(boundary_identifiers(mg)) do bId
                    ∂ₙv = map(boundary_grid(mg,bId),normal(mg,bId)) do x̄,n̂
                        n̂⋅∇v(x̄)
                    end
                    Hᵧ = inner_product(boundary_grid(mg,bId), stencil_set)
                    sum(Hᵧ*∂ₙv)
                end ≈ 2 atol=atol

            end
        end
        @test_broken false # Test that it calculates the right thing
    end
end
