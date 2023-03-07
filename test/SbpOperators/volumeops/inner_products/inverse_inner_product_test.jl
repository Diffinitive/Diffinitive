using Test

using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.LazyTensors

import Sbplib.SbpOperators.ConstantInteriorScalingOperator

@testset "Diagonal-stencil inverse_inner_product" begin
    Lx = π/2.
    Ly = Float64(π)
    g_1D = equidistant_grid(77, 0.0, Lx)
    g_2D = equidistant_grid((77,66), (0.0, 0.0), (Lx,Ly))
    @testset "inverse_inner_product" begin
        stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        @testset "0D" begin
            Hi = inverse_inner_product(ZeroDimGrid(1.), stencil_set)
            @test Hi isa LazyTensor{T,0,0} where T
        end
        @testset "1D" begin
            Hi = inverse_inner_product(g_1D, stencil_set)
            @test Hi isa LazyTensor{T,1,1} where T
        end
        @testset "2D" begin
            Hi = inverse_inner_product(g_2D, stencil_set)
            Hi_x = inverse_inner_product(g_2D.grids[1], stencil_set)
            Hi_y = inverse_inner_product(g_2D.grids[2], stencil_set)
            @test Hi == Hi_x⊗Hi_y
            @test Hi isa LazyTensor{T,2,2} where T
        end
    end

    @testset "Sizes" begin
        stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        @testset "1D" begin
            Hi = inverse_inner_product(g_1D, stencil_set)
            @test domain_size(Hi) == size(g_1D)
            @test range_size(Hi) == size(g_1D)
        end
        @testset "2D" begin
            Hi = inverse_inner_product(g_2D, stencil_set)
            @test domain_size(Hi) == size(g_2D)
            @test range_size(Hi) == size(g_2D)
        end
    end

    @testset "Accuracy" begin
        @testset "1D" begin
            v = eval_on(g_1D,x->sin(x))
            u = eval_on(g_1D,x->x^3-x^2+1)
            @testset "2nd order" begin
                stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=2)
                H = inner_product(g_1D, stencil_set)
                Hi = inverse_inner_product(g_1D, stencil_set)
                @test Hi*H*v ≈ v rtol = 1e-15
                @test Hi*H*u ≈ u rtol = 1e-15
            end
            @testset "4th order" begin
                stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)
                H = inner_product(g_1D, stencil_set)
                Hi = inverse_inner_product(g_1D, stencil_set)
                @test Hi*H*v ≈ v rtol = 1e-15
                @test Hi*H*u ≈ u rtol = 1e-15
            end
        end
        @testset "2D" begin
            v = eval_on(g_2D,(x,y)->sin(x)+cos(y))
            u = eval_on(g_2D,(x,y)->x*y + x^5 - sqrt(y))
            @testset "2nd order" begin
                stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=2)
                H = inner_product(g_2D, stencil_set)
                Hi = inverse_inner_product(g_2D, stencil_set)
                @test Hi*H*v ≈ v rtol = 1e-15
                @test Hi*H*u ≈ u rtol = 1e-15
            end
            @testset "4th order" begin
                stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)
                H = inner_product(g_2D, stencil_set)
                Hi = inverse_inner_product(g_2D, stencil_set)
                @test Hi*H*v ≈ v rtol = 1e-15
                @test Hi*H*u ≈ u rtol = 1e-15
            end
        end
    end
end
