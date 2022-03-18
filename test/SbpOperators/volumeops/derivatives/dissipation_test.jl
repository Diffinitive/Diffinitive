using Test

using Sbplib.SbpOperators
# using Sbplib.Grids
# using Sbplib.LazyTensors

using Sbplib.SbpOperators: Stencil

using Sbplib.SbpOperators: dissipation_interior_weights
using Sbplib.SbpOperators: dissipation_interior_stencil, dissipation_transpose_interior_stencil
using Sbplib.SbpOperators: midpoint, midpoint_transpose
using Sbplib.SbpOperators: dissipation_lower_closure_size, dissipation_upper_closure_size
using Sbplib.SbpOperators: dissipation_lower_closure_stencils,dissipation_upper_closure_stencils
using Sbplib.SbpOperators: dissipation_transpose_lower_closure_stencils, dissipation_transpose_upper_closure_stencils

@testset "dissipation_interior_weights" begin
    @test dissipation_interior_weights(1) == (-1, 1)
    @test dissipation_interior_weights(2) == (1,-2, 1)
    @test dissipation_interior_weights(3) == (-1, 3,-3, 1)
    @test dissipation_interior_weights(4) == (1, -4, 6, -4, 1)
end

@testset "dissipation_interior_stencil" begin
    @test dissipation_interior_stencil(1) == Stencil(-1,1, center=2)
    @test dissipation_interior_stencil(2) == Stencil(1,-2,1, center=2)
    @test dissipation_interior_stencil(3) == Stencil(-1,3,-3,1, center=3)
    @test dissipation_interior_stencil(4) == Stencil(1, -4, 6, -4, 1, center=3)
end

@testset "dissipation_transpose_interior_stencil" begin
    @test dissipation_transpose_interior_stencil(1) == Stencil(-1,1, center=1)
    @test dissipation_transpose_interior_stencil(2) == Stencil(1,-2,1, center=2)
    @test dissipation_transpose_interior_stencil(3) == Stencil(-1,3,-3,1, center=2)
    @test dissipation_transpose_interior_stencil(4) == Stencil(1, -4, 6, -4, 1, center=3)
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
            Stencil(-1,-1,    center=1),
            Stencil( 1, 1,-1, center=2),
        ),
        (1,-2,1) => (
            Stencil( 1, 1,    center=1),
            Stencil(-2,-2, 1,  center=2),
            Stencil( 1, 1,-2, 1, center=3),
        ),
        (-1,3,-3,1) => (
            Stencil(-1,-1,-1,          center=1),
            Stencil( 3, 3, 3,-1,       center=2),
            Stencil(-3,-3,-3, 3,-1,    center=3),
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
            Stencil(    1, center = 1),
        ),
        (1,-2,1) => (
            Stencil( 1, -2, 1, 1, center=2),
            Stencil(     1,-2,-2, center=2),
            Stencil(        1, 1, center=2),
        ),
        (-1,3,-3,1) => (
            Stencil( 1,-3, 3,-1,-1, center=2),
            Stencil(    1,-3, 3, 3, center=2),
            Stencil(       1,-3,-3, center=2),
            Stencil(          1, 1, center=2),
        ),
    )
    @testset "interior_weights = $w" for (w, closure_stencils) ∈ cases
        @test dissipation_transpose_upper_closure_stencils(w) == closure_stencils
    end
end
