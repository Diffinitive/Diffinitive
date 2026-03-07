using Test
using Diffinitive.Grids
using Diffinitive.SbpOperators
using StaticArrays

@testset "divergence" begin
    operator_path = sbp_operators_path()*"standard_diagonal.toml"
    stencil_set = read_stencil_set(operator_path; order=4)

    @testset "2D" begin
        g = equidistant_grid((0,0),(1,1), 20, 20)

        ∇̇ = divergence(g, stencil_set)

        v = map(x->@SVector[x[1], 0.], g)
        @test ∇̇*v ≈ map(x->1., g)

        v = map(x->@SVector[0.,x[1]], g)
        @test ∇̇*v ≈ map(x->0., g)

        v = map(x->@SVector[0., x[2]], g)
        @test ∇̇*v ≈ map(x->1., g)

        v = map(identity, g)
        @test ∇̇*v ≈ map(x->2., g)
    end
end


@testset "gradient" begin
    operator_path = sbp_operators_path()*"standard_diagonal.toml"
    stencil_set = read_stencil_set(operator_path; order=4)
    @testset "2D" begin
        g = equidistant_grid((0,0),(1,1), 20, 20)

        ∇ = gradient(g, stencil_set)

        v = map(x->x[1], g)
        @test ∇*v ≈ map(x->@SVector[1.,0], g)

        v = map(x->x[2], g)
        @test ∇*v ≈ map(x->@SVector[0,1.], g)

        v = map(x->x[1]*x[2], g)
        @test ∇*v ≈ map(x->@SVector[x[2],x[1]], g)

        v = map(x->sin(x[1]^2+x[2]^2), g)
        @test ∇*v ≈ map(x->cos(x[1]^2+x[2]^2)*@SVector[2x[1],2x[2]], g)
    end
end
