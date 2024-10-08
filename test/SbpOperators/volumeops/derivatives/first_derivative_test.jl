using Test


using Diffinitive.SbpOperators
using Diffinitive.Grids
using Diffinitive.LazyTensors

using Diffinitive.SbpOperators: closure_size, Stencil, VolumeOperator

@testset "first_derivative" begin
    @testset "Constructors" begin
        stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=2)

        g₁ = equidistant_grid(0., 1., 11)
        g₂ = equidistant_grid((0.,1.), (1.,3.), 11, 14)
        
        @test first_derivative(g₁, stencil_set) isa LazyTensor{Float64,1,1}
        @test first_derivative(g₂, stencil_set, 2) isa LazyTensor{Float64,2,2}

        interior_stencil = CenteredStencil(-1,0,1)
        closure_stencils = [Stencil(-1,1, center=1)]

        @test first_derivative(g₁, interior_stencil, closure_stencils) isa LazyTensor{Float64,1,1}
    end

    @testset "Accuracy conditions" begin
        N = 20
        g = equidistant_grid(0//1, 2//1, N)

        monomial(x,k) = k < 0 ? zero(x) : x^k/factorial(k)
        @testset for order ∈ [2,4]
            stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order)
            D₁ = first_derivative(g, stencil_set)

            @testset "boundary x^$k" for k ∈ 0:order÷2
                v = eval_on(g, x->monomial(x,k))

                @testset for i ∈ 1:closure_size(D₁)
                    x, = g[i]
                    @test (D₁*v)[i] == monomial(x,k-1)
                end

                @testset for i ∈ (N-closure_size(D₁)+1):N
                    x, = g[i]
                    @test (D₁*v)[i] == monomial(x,k-1)
                end
            end

            @testset "interior x^$k" for k ∈ 0:order
                v = eval_on(g, x->monomial(x,k))

                x, = g[10]
                @test (D₁*v)[10] == monomial(x,k-1)
            end
        end
    end

    @testset "Accuracy on function" begin
        @testset "1D" begin
            g = equidistant_grid(0., 1., 30)
            v = eval_on(g, x->exp(x))
            @testset for (order, tol) ∈ [(2, 6e-3),(4, 2e-4)]
                stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order)
                D₁ = first_derivative(g, stencil_set)

                @test D₁*v ≈ v rtol=tol
            end
        end

        @testset "2D" begin
            g = equidistant_grid((0.,0.),(1.,2.), 30, 60)
            v = eval_on(g, (x,y)->exp(0.8x+1.2*y))
            @testset for (order, tol) ∈ [(2, 6e-3),(4, 3e-4)]
                stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order)
                Dx = first_derivative(g, stencil_set, 1)
                Dy = first_derivative(g, stencil_set, 2)

                @test Dx*v ≈ 0.8v rtol=tol
                @test Dy*v ≈ 1.2v rtol=tol
            end
        end
    end
end

