using Diffinitive.Grids
using Diffinitive.Grids: Line
using StaticArrays

@testset "Line" begin
    @testset "Constructors" begin
        @test Line([1,2],[2,3]) isa Line{SVector{2,Int}}
        @test Line((1,2),(2,3)) isa Line{SVector{2,Int}}
        @test Line(@SVector[1,2],[2,3]) isa Line{SVector{2,Int}}
        @test Line(@SVector[1,2],@SVector[2,3]) isa Line{SVector{2,Int}}

        @test Line([1,2],[2.,3]) isa Line{SVector{2,Float64}}
        @test Line(@SVector[1,2.],@SVector[2,3]) isa Line{SVector{2,Float64}}
        @test Line((1,2.),(2,3)) isa Line{SVector{2,Float64}}
    end

    @testset "Evaluation" begin
        l = Line([1,2],[2,3])

        @test l(0) == [1,2]
        @test l(1) == [1,2] + [2,3]
        @test l(1/2) == [1,2] + [2,3]/2
    end
end

@testset "LineSegment" begin
    @testset "Constructors" begin
    end

    @test_broken false
end

@testset "linesegments" begin
    @testset "Constructors" begin
    end

    @test_broken false
end

@testset "polygon_edges" begin
    @testset "Constructors" begin
    end

    @test_broken false
end

@testset "Circle" begin
    @testset "Constructors" begin
    end

    @test_broken false
end

@testset "TransfiniteInterpolationSurface" begin
    @testset "Constructors" begin
    end

    @test_broken false
end
