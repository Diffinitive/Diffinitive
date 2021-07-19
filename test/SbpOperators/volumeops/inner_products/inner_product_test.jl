using Test

using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.LazyTensors


@testset "Diagonal-stencil inner_product" begin
    Lx = π/2.
    Ly = Float64(π)
    Lz = 1.
    g_1D = EquidistantGrid(77, 0.0, Lx)
    g_2D = EquidistantGrid((77,66), (0.0, 0.0), (Lx,Ly))
    g_3D = EquidistantGrid((10,10, 10), (0.0, 0.0, 0.0), (Lx,Ly,Lz))
    integral(H,v) = sum(H*v)
    @testset "inner_product" begin
        op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        @testset "0D" begin
            H = inner_product(EquidistantGrid{Float64}(), op.quadratureClosure, CenteredStencil(1.))
            @test H == IdentityMapping{Float64}()
            @test H isa TensorMapping{T,0,0} where T
        end
        @testset "1D" begin
            H = inner_product(g_1D, op.quadratureClosure, CenteredStencil(1.))
            @test H == inner_product(g_1D, op.quadratureClosure, CenteredStencil(1.))
            @test H isa TensorMapping{T,1,1} where T
        end
        @testset "2D" begin
            H = inner_product(g_2D, op.quadratureClosure, CenteredStencil(1.))
            H_x = inner_product(restrict( g_2D,1),op.quadratureClosure, CenteredStencil(1.))
            H_y = inner_product(restrict( g_2D,2),op.quadratureClosure, CenteredStencil(1.))
            @test H == H_x⊗H_y
            @test H isa TensorMapping{T,2,2} where T
        end
    end

    @testset "Sizes" begin
        op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        @testset "1D" begin
            H = inner_product(g_1D, op.quadratureClosure, CenteredStencil(1.))
            @test domain_size(H) == size(g_1D)
            @test range_size(H) == size(g_1D)
        end
        @testset "2D" begin
            H = inner_product(g_2D, op.quadratureClosure, CenteredStencil(1.))
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
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=2)
                H = inner_product(g_1D, op.quadratureClosure, CenteredStencil(1.))
                for i = 1:2
                    @test integral(H,v[i]) ≈ v[i+1][end] - v[i+1][1] rtol = 1e-14
                end
                @test integral(H,u) ≈ 1. rtol = 1e-4
            end

            @testset "4th order" begin
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
                H = inner_product(g_1D, op.quadratureClosure, CenteredStencil(1.))
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
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=2)
                H = inner_product(g_2D, op.quadratureClosure, CenteredStencil(1.))
                @test integral(H,v) ≈ b*Lx*Ly rtol = 1e-13
                @test integral(H,u) ≈ π rtol = 1e-4
            end
            @testset "4th order" begin
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
                H = inner_product(g_2D, op.quadratureClosure, CenteredStencil(1.))
                @test integral(H,v) ≈ b*Lx*Ly rtol = 1e-13
                @test integral(H,u) ≈ π rtol = 1e-8
            end
        end
    end
end
