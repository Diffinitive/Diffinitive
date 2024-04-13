using Test

using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.LazyTensors

import Sbplib.SbpOperators.ConstantInteriorScalingOperator

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
end
