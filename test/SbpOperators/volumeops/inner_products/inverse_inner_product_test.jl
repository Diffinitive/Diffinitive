using Test

using Diffinitive.SbpOperators
using Diffinitive.Grids
using Diffinitive.LazyTensors

import Diffinitive.SbpOperators.ConstantInteriorScalingOperator

using StaticArrays

@testset "Diagonal-stencil inverse_inner_product" begin
    Lx = π/2.
    Ly = Float64(π)
    g_1D = equidistant_grid(0.0, Lx, 77)
    g_2D = equidistant_grid((0.0, 0.0), (Lx,Ly), 77, 66)
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

    @testset "MappedGrid" begin
        stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        c = Chart(unitsquare()) do (ξ,η)
            @SVector[2ξ + η*(1-η), 3η+(1+η/2)*ξ^2]
        end
        Grids.jacobian(c::typeof(c), (ξ,η)) = @SMatrix[2 1-2η; (2+η)*ξ 3+ξ^2/2]

        mg = equidistant_grid(c, 10,13)

        @test inverse_inner_product(mg, stencil_set) isa LazyTensor{<:Any, 2,2}
        @test_broken false # Test that it calculates the right thing
    end
end
