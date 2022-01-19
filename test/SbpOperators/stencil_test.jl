using Test
using Sbplib.SbpOperators
import Sbplib.SbpOperators.Stencil
import Sbplib.SbpOperators.NestedStencil
import Sbplib.SbpOperators.scale

@testset "Stencil" begin
    s = Stencil((-2,2), (1.,2.,2.,3.,4.))
    @test s isa Stencil{Float64, 5}

    @test eltype(s) == Float64
    @test SbpOperators.scale(s, 2) == Stencil((-2,2), (2.,4.,4.,6.,8.))

    @test Stencil(1,2,3,4; center=1) == Stencil((0, 3),(1,2,3,4))
    @test Stencil(1,2,3,4; center=2) == Stencil((-1, 2),(1,2,3,4))
    @test Stencil(1,2,3,4; center=4) == Stencil((-3, 0),(1,2,3,4))

    @test CenteredStencil(1,2,3,4,5) == Stencil((-2, 2), (1,2,3,4,5))
    @test_throws ArgumentError CenteredStencil(1,2,3,4)

    # Changing the type of the weights
    @test Stencil{Float64}(Stencil(1,2,3,4,5; center=2)) == Stencil(1.,2.,3.,4.,5.; center=2)
    @test Stencil{Float64}(CenteredStencil(1,2,3,4,5)) == CenteredStencil(1.,2.,3.,4.,5.)
    @test Stencil{Int}(Stencil(1.,2.,3.,4.,5.; center=2)) == Stencil(1,2,3,4,5; center=2)
    @test Stencil{Rational}(Stencil(1.,2.,3.,4.,5.; center=2)) == Stencil(1//1,2//1,3//1,4//1,5//1; center=2)

    @testset "convert" begin
        @test convert(Stencil{Float64}, Stencil(1,2,3,4,5; center=2)) == Stencil(1.,2.,3.,4.,5.; center=2)
        @test convert(Stencil{Float64}, CenteredStencil(1,2,3,4,5)) == CenteredStencil(1.,2.,3.,4.,5.)
        @test convert(Stencil{Int}, Stencil(1.,2.,3.,4.,5.; center=2)) == Stencil(1,2,3,4,5; center=2)
        @test convert(Stencil{Rational}, Stencil(1.,2.,3.,4.,5.; center=2)) == Stencil(1//1,2//1,3//1,4//1,5//1; center=2)
    end
end

@testset "NestedStencil" begin

    @testset "Constructors" begin
        s1 = Stencil(-1, 1, 0; center = 1)
        s2 = Stencil(-1, 0, 1; center = 2)
        s3 = Stencil( 0,-1, 1; center = 3)

        ns = NestedStencil(CenteredStencil(s1,s2,s3))
        @test ns isa NestedStencil{Int,3}

        @test CenteredNestedStencil(s1,s2,s3) == ns

        @test NestedStencil(s1,s2,s3, center = 2) == ns
        @test NestedStencil(s1,s2,s3, center = 1) == NestedStencil(Stencil(s1,s2,s3, center=1))

        @test NestedStencil((-1,1,0),(-1,0,1),(0,-1,1), center=2) == ns


        @testset "Error handling" begin

        end
    end
end
