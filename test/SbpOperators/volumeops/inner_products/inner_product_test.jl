using Test

using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.LazyTensors

import Sbplib.SbpOperators.ConstantInteriorScalingOperator

@testset "Diagonal-stencil inner_product" begin
    Lx = π/2.
    Ly = Float64(π)
    Lz = 1.
    g_1D = EquidistantGrid(77, 0.0, Lx)
    g_2D = EquidistantGrid((77,66), (0.0, 0.0), (Lx,Ly))
    g_3D = EquidistantGrid((10,10, 10), (0.0, 0.0, 0.0), (Lx,Ly,Lz))
    integral(H,v) = sum(H*v)
    @testset "inner_product" begin
        stencil_set = StencilSet(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        quadrature_interior = parse_scalar(stencil_set["H"]["inner"])
        quadrature_closure = parse_tuple(stencil_set["H"]["closure"])
        @testset "0D" begin
            H = inner_product(EquidistantGrid{Float64}(), quadrature_interior, quadrature_closure)
            @test H == inner_product(EquidistantGrid{Float64}(), stencil_set)
            @test H == IdentityMapping{Float64}()
            @test H isa TensorMapping{T,0,0} where T
        end
        @testset "1D" begin
            H = inner_product(g_1D, quadrature_interior, quadrature_closure)
            @test H == inner_product(g_1D, stencil_set)
            @test H isa ConstantInteriorScalingOperator
            @test H isa TensorMapping{T,1,1} where T
        end
        @testset "2D" begin
            H = inner_product(g_2D, quadrature_interior, quadrature_closure)
            H_x = inner_product(restrict(g_2D,1), quadrature_interior, quadrature_closure)
            H_y = inner_product(restrict(g_2D,2), quadrature_interior, quadrature_closure)
            @test H == inner_product(g_2D, stencil_set)
            @test H == H_x⊗H_y
            @test H isa TensorMapping{T,2,2} where T
        end
    end

    @testset "Sizes" begin
        stencil_set = StencilSet(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        quadrature_interior = parse_scalar(stencil_set["H"]["inner"])
        quadrature_closure = parse_tuple(stencil_set["H"]["closure"])
        @testset "1D" begin
            H = inner_product(g_1D, quadrature_interior, quadrature_closure)
            @test domain_size(H) == size(g_1D)
            @test range_size(H) == size(g_1D)
        end
        @testset "2D" begin
            H = inner_product(g_2D, quadrature_interior, quadrature_closure)
            @test domain_size(H) == size(g_2D)
            @test range_size(H) == size(g_2D)
        end
    end

    @testset "Accuracy" begin
        @testset "1D" begin
            v = ()
            for i = 0:4
                f_i(x) = 1/factorial(i)*x^i
                v = (v...,evalOn(g_1D,f_i))
            end
            u = evalOn(g_1D,x->sin(x))

            @testset "2nd order" begin
                stencil_set = StencilSet(sbp_operators_path()*"standard_diagonal.toml"; order=2)
                quadrature_interior = parse_scalar(stencil_set["H"]["inner"])
                quadrature_closure = parse_tuple(stencil_set["H"]["closure"])
                H = inner_product(g_1D, quadrature_interior, quadrature_closure)
                for i = 1:2
                    @test integral(H,v[i]) ≈ v[i+1][end] - v[i+1][1] rtol = 1e-14
                end
                @test integral(H,u) ≈ 1. rtol = 1e-4
            end

            @testset "4th order" begin
                stencil_set = StencilSet(sbp_operators_path()*"standard_diagonal.toml"; order=4)
                quadrature_interior = parse_scalar(stencil_set["H"]["inner"])
                quadrature_closure = parse_tuple(stencil_set["H"]["closure"])
                H = inner_product(g_1D, quadrature_interior, quadrature_closure)
                for i = 1:4
                    @test integral(H,v[i]) ≈ v[i+1][end] -  v[i+1][1] rtol = 1e-14
                end
                @test integral(H,u) ≈ 1. rtol = 1e-8
            end
        end

        @testset "2D" begin
            b = 2.1
            v = b*ones(Float64, size(g_2D))
            u = evalOn(g_2D,(x,y)->sin(x)+cos(y))
            @testset "2nd order" begin
                stencil_set = StencilSet(sbp_operators_path()*"standard_diagonal.toml"; order=2)
                quadrature_interior = parse_scalar(stencil_set["H"]["inner"])
                quadrature_closure = parse_tuple(stencil_set["H"]["closure"])
                H = inner_product(g_2D, quadrature_interior, quadrature_closure)
                @test integral(H,v) ≈ b*Lx*Ly rtol = 1e-13
                @test integral(H,u) ≈ π rtol = 1e-4
            end
            @testset "4th order" begin
                stencil_set = StencilSet(sbp_operators_path()*"standard_diagonal.toml"; order=4)
                quadrature_interior = parse_scalar(stencil_set["H"]["inner"])
                quadrature_closure = parse_tuple(stencil_set["H"]["closure"])
                H = inner_product(g_2D, quadrature_interior, quadrature_closure)
                @test integral(H,v) ≈ b*Lx*Ly rtol = 1e-13
                @test integral(H,u) ≈ π rtol = 1e-8
            end
        end
    end
end
