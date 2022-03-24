using Test

using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.LazyTensors

using Sbplib.SbpOperators: Stencil

using Sbplib.SbpOperators: dissipation_interior_weights
using Sbplib.SbpOperators: dissipation_interior_stencil, dissipation_transpose_interior_stencil
using Sbplib.SbpOperators: midpoint, midpoint_transpose
using Sbplib.SbpOperators: dissipation_lower_closure_size, dissipation_upper_closure_size
using Sbplib.SbpOperators: dissipation_lower_closure_stencils,dissipation_upper_closure_stencils
using Sbplib.SbpOperators: dissipation_transpose_lower_closure_stencils, dissipation_transpose_upper_closure_stencils

"""
    monomial(x,k)

Evaluates ``x^k/k!` with the convetion that it is ``0`` for all ``k<0``.
Has the property that ``d/dx monomial(x,k) = monomial(x,k-1)``
"""
function monomial(x,k)
    if k < 0
        return zero(x)
    end
    x^k/factorial(k)
end

@testset "undivided_dissipation" begin
    g = EquidistantGrid(20, 0., 11.)
    D,Dᵀ = undivided_dissipation(g, 1)

    @test D isa LazyTensor{Float64,1,1} where T
    @test Dᵀ isa LazyTensor{Float64,1,1} where T

     @testset "Accuracy conditions" begin
        N = 20
        g = EquidistantGrid(N, 0//1,2//1)
        h = only(spacing(g))
        @testset "D_$p" for p ∈ [1,2,3,4]
            D,Dᵀ = undivided_dissipation(g, p)

            @testset "x^$k" for k ∈ 0:p
                v  = evalOn(g, x->monomial(x,k))
                vₚₓ = evalOn(g, x->monomial(x,k-p))

                @test D*v == h^p * vₚₓ
            end
        end
    end

    @testset "tanspose equality" begin
        function get_matrix(D)
            N = only(range_size(D))
            M = only(domain_size(D))

            Dmat = zeros(N,M)
            e = zeros(M)
            for i ∈ 1:M
                if i > 1
                    e[i-1] = 0.
                end
                e[i] = 1.
                Dmat[:,i] = D*e
            end

            return Dmat
        end

        g = EquidistantGrid(11, 0., 1.)
        @testset "D_$p" for p ∈ [1,2,3,4]
            D,Dᵀ = undivided_dissipation(g, p)

            D̄  = get_matrix(D)
            D̄ᵀ = get_matrix(Dᵀ)

            @test D̄ == D̄ᵀ'
        end
    end
end

@testset "dissipation_interior_weights" begin
    @test dissipation_interior_weights(1) == (-1, 1)
    @test dissipation_interior_weights(2) == (1,-2, 1)
    @test dissipation_interior_weights(3) == (-1, 3,-3, 1)
    @test dissipation_interior_weights(4) == (1, -4, 6, -4, 1)
end

@testset "dissipation_interior_stencil" begin
    @test dissipation_interior_stencil(dissipation_interior_weights(1)) == Stencil(-1, 1, center=2)
    @test dissipation_interior_stencil(dissipation_interior_weights(2)) == Stencil( 1,-2, 1, center=2)
    @test dissipation_interior_stencil(dissipation_interior_weights(3)) == Stencil(-1, 3,-3, 1, center=3)
    @test dissipation_interior_stencil(dissipation_interior_weights(4)) == Stencil( 1,-4, 6,-4, 1, center=3)
end

@testset "dissipation_transpose_interior_stencil" begin
    @test dissipation_transpose_interior_stencil(dissipation_interior_weights(1)) == Stencil(1,-1, center=1)
    @test dissipation_transpose_interior_stencil(dissipation_interior_weights(2)) == Stencil(1,-2, 1, center=2)
    @test dissipation_transpose_interior_stencil(dissipation_interior_weights(3)) == Stencil(1,-3, 3,-1, center=2)
    @test dissipation_transpose_interior_stencil(dissipation_interior_weights(4)) == Stencil(1,-4, 6,-4, 1, center=3)
end

@testset "midpoint" begin
    @test midpoint((1,1)) == 2
    @test midpoint((1,1,1)) == 2
    @test midpoint((1,1,1,1)) == 3
    @test midpoint((1,1,1,1,1)) == 3
end

@testset "midpoint_transpose" begin
    @test midpoint_transpose((1,1)) == 1
    @test midpoint_transpose((1,1,1)) == 2
    @test midpoint_transpose((1,1,1,1)) == 2
    @test midpoint_transpose((1,1,1,1,1)) == 3
end

@testset "dissipation_lower_closure_size" begin
    @test dissipation_lower_closure_size((1,1)) == 1
    @test dissipation_lower_closure_size((1,1,1)) == 1
    @test dissipation_lower_closure_size((1,1,1,1)) == 2
    @test dissipation_lower_closure_size((1,1,1,1,1)) == 2
end

@testset "dissipation_upper_closure_size" begin
    @test dissipation_upper_closure_size((1,1)) == 0
    @test dissipation_upper_closure_size((1,1,1)) == 1
    @test dissipation_upper_closure_size((1,1,1,1)) == 1
    @test dissipation_upper_closure_size((1,1,1,1,1)) == 2
end

@testset "dissipation_lower_closure_stencils" begin
    cases = (
        (-1,1) => (
            Stencil(-1, 1, center=1),
        ),
        (1,-2,1) => (
            Stencil( 1,-2, 1, center=1),
        ),
        (-1,3,-3,1) => (
            Stencil(-1,3,-3,1, center=1),
            Stencil(-1,3,-3,1, center=2),
        ),
        (1, -4, 6, -4, 1) => (
            Stencil(1, -4, 6, -4, 1, center=1),
            Stencil(1, -4, 6, -4, 1, center=2),
        )
    )
    @testset "interior_weights = $w" for (w, closure_stencils) ∈ cases
        @test dissipation_lower_closure_stencils(w) == closure_stencils
    end
end

@testset "dissipation_upper_closure_stencils" begin
    cases = (
        (-1,1) => (),
        (1,-2,1) => (
            Stencil( 1,-2, 1, center=3),
        ),
        (-1,3,-3,1) => (
            Stencil(-1,3,-3,1, center=4),
        ),
        (1, -4, 6, -4, 1) => (
            Stencil(1, -4, 6, -4, 1, center=4),
            Stencil(1, -4, 6, -4, 1, center=5),
        )
    )
    @testset "interior_weights = $w" for (w, closure_stencils) ∈ cases
        @test dissipation_upper_closure_stencils(w) == closure_stencils
    end
end


@testset "dissipation_transpose_lower_closure_stencils" begin
    cases = (
        (-1,1) => (
            Stencil(-1,-1, 0, center=1),
            Stencil( 1, 1,-1, center=2),
        ),
        (1,-2,1) => (
            Stencil( 1, 1, 0, 0, center=1),
            Stencil(-2,-2, 1, 0, center=2),
            Stencil( 1, 1,-2, 1, center=3),
        ),
        (-1,3,-3,1) => (
            Stencil(-1,-1,-1, 0, 0, 0, center=1),
            Stencil( 3, 3, 3,-1, 0, 0, center=2),
            Stencil(-3,-3,-3, 3,-1, 0, center=3),
            Stencil( 1, 1, 1,-3, 3,-1, center=4),
        ),
    )
    @testset "interior_weights = $w" for (w, closure_stencils) ∈ cases
        @test dissipation_transpose_lower_closure_stencils(w) == closure_stencils
    end
end

@testset "dissipation_transpose_upper_closure_stencils" begin
    cases = (
        (-1,1) => (
            Stencil( 1,-1, center = 1),
            Stencil( 0, 1, center = 2),
        ),
        (1,-2,1) => (
            Stencil( 1,-2, 1, 1, center=2),
            Stencil( 0, 1,-2,-2, center=3),
            Stencil( 0, 0, 1, 1, center=4),
        ),
        (-1,3,-3,1) => (
            Stencil( 1,-3, 3,-1,-1, center=2),
            Stencil( 0, 1,-3, 3, 3, center=3),
            Stencil( 0, 0, 1,-3,-3, center=4),
            Stencil( 0, 0, 0, 1, 1, center=5),
        ),
    )
    @testset "interior_weights = $w" for (w, closure_stencils) ∈ cases
        @test dissipation_transpose_upper_closure_stencils(w) == closure_stencils
    end
end
