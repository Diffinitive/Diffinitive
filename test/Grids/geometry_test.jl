using Diffinitive.Grids
using Diffinitive.Grids: Line, LineSegment, linesegments, polygon_edges, Circle
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
        @test LineSegment([1,2],[2,3]) isa LineSegment{SVector{2,Int}}
        @test LineSegment((1,2),(2,3)) isa LineSegment{SVector{2,Int}}
        @test LineSegment(@SVector[1,2],[2,3]) isa LineSegment{SVector{2,Int}}
        @test LineSegment(@SVector[1,2],@SVector[2,3]) isa LineSegment{SVector{2,Int}}

        @test LineSegment([1,2],[2.,3]) isa LineSegment{SVector{2,Float64}}
        @test LineSegment(@SVector[1,2.],@SVector[2,3]) isa LineSegment{SVector{2,Float64}}
        @test LineSegment((1,2.),(2,3)) isa LineSegment{SVector{2,Float64}}
    end

    @testset "Evaluation" begin
        l = LineSegment([1,2],[2,3])

        @test l(0) == [1,2]
        @test l(1) == [2,3]
        @test l(1/2) == [1,2]/2 + [2,3]/2
    end
end

@testset "linesegments" begin
    a,b,c,d = [1,1],[2,2],[3,3],[4,4]
    @test linesegments(a,b) == [
        LineSegment(a,b),
    ]

    @test linesegments(a,b,c) == [
        LineSegment(a,b),
        LineSegment(b,c),
    ]

    @test linesegments(a,b,c,d) == [
        LineSegment(a,b),
        LineSegment(b,c),
        LineSegment(c,d),
    ]
end

@testset "polygon_edges" begin
    a,b,c,d = [1,1],[2,2],[3,3],[4,4]
    @test polygon_edges(a,b) == [
        LineSegment(a,b),
        LineSegment(b,a),
    ]

    @test polygon_edges(a,b,c) == [
        LineSegment(a,b),
        LineSegment(b,c),
        LineSegment(c,a),
    ]

    @test polygon_edges(a,b,c,d) == [
        LineSegment(a,b),
        LineSegment(b,c),
        LineSegment(c,d),
        LineSegment(d,a),
    ]
end

@testset "Circle" begin
    @testset "Constructors" begin
        @test Circle([1,2], 1) isa Circle{SVector{2,Int},Int}
        @test Circle([1,2], 1.) isa Circle{SVector{2,Int},Float64}
        @test Circle([1,2.], 1.) isa Circle{SVector{2,Float64},Float64}
        @test Circle([1,2.], 1) isa Circle{SVector{2,Float64},Int}
        @test Circle((1,2.), 1.) isa Circle{SVector{2,Float64},Float64}
        @test Circle((1,2), 1.) isa Circle{SVector{2,Int},Float64}
        @test Circle((1.,2), 1) isa Circle{SVector{2,Float64},Int}
        @test Circle((1,2), 1) isa Circle{SVector{2,Int},Int}
        @test Circle(@SVector[1,2], 1.) isa Circle{SVector{2,Int},Float64}
        @test Circle(@SVector[1,2.], 1.) isa Circle{SVector{2,Float64},Float64}
    end

    @testset "Evaluation" begin
        c = Circle([0,0], 1)
        @test c(0) ≈ [1,0]
        @test c(π/2) ≈ [0,1]
        @test c(π) ≈ [-1,0]
        @test c(3π/2) ≈ [0,-1]
        @test c(π/4) ≈ [1/√(2),1/√(2)]

        c = Circle([0,0], 2)
        @test c(0) ≈ [2,0]
        @test c(π/2) ≈ [0,2]
        @test c(π) ≈ [-2,0]
        @test c(3π/2) ≈ [0,-2]
        @test c(π/4) ≈ [√(2),√(2)]
    end
end

@testset "TransfiniteInterpolationSurface" begin
    @testset "Constructors" begin
    end

    @test_broken false
end
