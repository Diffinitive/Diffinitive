using Test
using Sbplib.SbpOperators
import Sbplib.SbpOperators.Stencil
import Sbplib.SbpOperators.NestedStencil
import Sbplib.SbpOperators.scale

@testset "Stencil" begin
    s = Stencil(-2:2, (1.,2.,2.,3.,4.))
    @test s isa Stencil{Float64, 5}

    @test eltype(s) == Float64

    @test length(s) == 5
    @test length(Stencil(-1:2, (1,2,3,4))) == 4

    @test SbpOperators.scale(s, 2) == Stencil(-2:2, (2.,4.,4.,6.,8.))

    @test Stencil(1,2,3,4; center=1) == Stencil(0:3,(1,2,3,4))
    @test Stencil(1,2,3,4; center=2) == Stencil(-1:2,(1,2,3,4))
    @test Stencil(1,2,3,4; center=4) == Stencil(-3:0,(1,2,3,4))

    @test CenteredStencil(1,2,3,4,5) == Stencil(-2:2, (1,2,3,4,5))
    @test_throws ArgumentError CenteredStencil(1,2,3,4)

    # Changing the type of the weights
    @test Stencil{Float64}(Stencil(1,2,3,4,5; center=2)) == Stencil(1.,2.,3.,4.,5.; center=2)
    @test Stencil{Float64}(CenteredStencil(1,2,3,4,5)) == CenteredStencil(1.,2.,3.,4.,5.)
    @test Stencil{Int}(Stencil(1.,2.,3.,4.,5.; center=2)) == Stencil(1,2,3,4,5; center=2)
    @test Stencil{Rational}(Stencil(1.,2.,3.,4.,5.; center=2)) == Stencil(1//1,2//1,3//1,4//1,5//1; center=2)

    @testset "convert" begin
        @test convert(Stencil{Float64}, Stencil(1,2,3,4,5; center=2)) == Stencil(1.,2.,3.,4.,5.; center=2)
        @test convert(Stencil{Float64,5}, CenteredStencil(1,2,3,4,5)) == CenteredStencil(1.,2.,3.,4.,5.)
        @test convert(Stencil{Int,5}, Stencil(1.,2.,3.,4.,5.; center=2)) == Stencil(1,2,3,4,5; center=2)
        @test convert(Stencil{Rational,5}, Stencil(1.,2.,3.,4.,5.; center=2)) == Stencil(1//1,2//1,3//1,4//1,5//1; center=2)
    end

    @testset "promotion of weights" begin
        @test Stencil(1.,2; center = 1) isa Stencil{Float64, 2}
        @test Stencil(1,2//2; center = 1) isa Stencil{Rational{Int64}, 2}
    end

    @testset "promotion" begin
        @test promote(Stencil(1,1;center=1), Stencil(2.,2.;center=2)) == (Stencil(1.,1.;center=1), Stencil(2.,2.;center=2))
    end

    @testset "type stability" begin
        s_int = CenteredStencil(1,2,3)
        s_float = CenteredStencil(1.,2.,3.)
        v_int = rand(1:10,10);
        v_float = rand(10);

        @inferred SbpOperators.apply_stencil(s_int, v_int, 2)
        @inferred SbpOperators.apply_stencil(s_float, v_float, 2)
        @inferred SbpOperators.apply_stencil(s_int,  v_float, 2)
        @inferred SbpOperators.apply_stencil(s_float, v_int, 2)

        # TODO: apply backwards
    end
end

@testset "NestedStencil" begin

    @testset "Constructors" begin
        s1 = CenteredStencil(-1, 1, 0)
        s2 = CenteredStencil(-1, 0, 1)
        s3 = CenteredStencil( 0,-1, 1)

        ns = NestedStencil(CenteredStencil(s1,s2,s3))
        @test ns isa NestedStencil{Int,3}

        @test CenteredNestedStencil(s1,s2,s3) == ns

        @test NestedStencil(s1,s2,s3, center = 2) == ns
        @test NestedStencil(s1,s2,s3, center = 1) == NestedStencil(Stencil(s1,s2,s3, center=1))

        @test NestedStencil((-1,1,0),(-1,0,1),(0,-1,1), center=2) == ns
        @test CenteredNestedStencil((-1,1,0),(-1,0,1),(0,-1,1)) == ns
        @test NestedStencil((-1,1,0),(-1,0,1),(0,-1,1), center=1) == NestedStencil(Stencil(
            Stencil(-1, 1, 0; center=1),
            Stencil(-1, 0, 1; center=1),
            Stencil( 0,-1, 1; center=1);
            center=1
        ))

        @testset "Error handling" begin
        end
    end

    @testset "scale" begin
        ns = NestedStencil((-1,1,0),(-1,0,1),(0,-1,1), center=2)
        @test SbpOperators.scale(ns, 2) == NestedStencil((-2,2,0),(-2,0,2),(0,-2,2), center=2)
    end

    @testset "conversion" begin
        ns = NestedStencil((-1,1,0),(-1,0,1),(0,-1,1), center=2)
        @test NestedStencil{Float64}(ns) == NestedStencil((-1.,1.,0.),(-1.,0.,1.),(0.,-1.,1.), center=2)
        @test NestedStencil{Rational}(ns) == NestedStencil((-1//1,1//1,0//1),(-1//1,0//1,1//1),(0//1,-1//1,1//1), center=2)

        @test convert(NestedStencil{Float64}, ns) == NestedStencil((-1.,1.,0.),(-1.,0.,1.),(0.,-1.,1.), center=2)
        @test convert(NestedStencil{Rational}, ns) == NestedStencil((-1//1,1//1,0//1),(-1//1,0//1,1//1),(0//1,-1//1,1//1), center=2)
    end

    @testset "promotion of weights" begin
        @test NestedStencil((-1,1,0),(-1.,0.,1.),(0,-1,1), center=2) isa NestedStencil{Float64,3,3}
        @test NestedStencil((-1,1,0),(-1,0,1),(0//1,-1,1), center=2) isa NestedStencil{Rational{Int64},3,3}
    end

    @testset "promotion" begin
        promote(
            CenteredNestedStencil((-1,1,0),(-1,0,1),(0,-1,1)),
            CenteredNestedStencil((-1.,1.,0.),(-1.,0.,1.),(0.,-1.,1.))
        ) == (
            CenteredNestedStencil((-1.,1.,0.),(-1.,0.,1.),(0.,-1.,1.)),
            CenteredNestedStencil((-1.,1.,0.),(-1.,0.,1.),(0.,-1.,1.))
        )
    end

    @testset "apply" begin
        c = [  1,  3,  6, 10, 15, 21, 28, 36, 45, 55]
        v = [  2,  3,  5,  7, 11, 13, 17, 19, 23, 29]

        # Centered
        ns = NestedStencil((-1,1,0),(-1,0,1),(0,-2,2), center=2)
        @test SbpOperators.apply_inner_stencils(ns, c, 4) == Stencil(4,9,10; center=2)
        @test SbpOperators.apply_inner_stencils_backwards(ns, c, 4) == Stencil(-5,-9,-8; center=2)

        @test SbpOperators.apply_stencil(ns, c, v, 4) == 4*5 + 9*7 + 10*11
        @test SbpOperators.apply_stencil_backwards(ns, c, v, 4) == -8*5 - 9*7 - 5*11

        # Non-centered
        ns = NestedStencil((-1,1,0),(-1,0,1),(0,-1,1), center=1)
        @test SbpOperators.apply_inner_stencils(ns, c, 4) == Stencil(5,11,6; center=1)
        @test SbpOperators.apply_inner_stencils_backwards(ns, c, 4) == Stencil(-4,-7,-3; center=1)

        @test SbpOperators.apply_stencil(ns, c, v, 4) == 5*7 + 11*11 + 6*13
        @test SbpOperators.apply_stencil_backwards(ns, c, v, 4) == -3*3 - 7*5 - 4*7
    end

    @testset "type stability" begin
        s_int = CenteredNestedStencil((1,2,3),(1,2,3),(1,2,3))
        s_float = CenteredNestedStencil((1.,2.,3.),(1.,2.,3.),(1.,2.,3.))

        v_int = rand(1:10,10);
        v_float = rand(10);

        c_int = rand(1:10,10);
        c_float = rand(10);

        @inferred SbpOperators.apply_stencil(s_int,   c_int, v_int,   2)
        @inferred SbpOperators.apply_stencil(s_float, c_int, v_float, 2)
        @inferred SbpOperators.apply_stencil(s_int,   c_int, v_float, 2)
        @inferred SbpOperators.apply_stencil(s_float, c_int, v_int,   2)

        @inferred SbpOperators.apply_stencil(s_int,   c_float, v_int,   2)
        @inferred SbpOperators.apply_stencil(s_float, c_float, v_float, 2)
        @inferred SbpOperators.apply_stencil(s_int,   c_float, v_float, 2)
        @inferred SbpOperators.apply_stencil(s_float, c_float, v_int,   2)

        # TODO: apply backwards
    end

end
