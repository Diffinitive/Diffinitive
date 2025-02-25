using Diffinitive.Grids
using Diffinitive.Grids: Line, LineSegment, linesegments, polygon_edges, Circle, TransfiniteInterpolationSurface, check_transfiniteinterpolation
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

    @testset "Grids.jacobian" begin
        l = Line([1,2],[2,3])

        @test Grids.jacobian(l,0) == [2,3]
        @test Grids.jacobian(l,1) == [2,3]
        @test Grids.jacobian(l,1/2) == [2,3]
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

    @testset "Grids.jacobian" begin
        a, b = [1,2], [2,3]
        l = LineSegment(a,b)
        d = b-a

        @test Grids.jacobian(l,0) == d
        @test Grids.jacobian(l,1) == d
        @test Grids.jacobian(l,1/2) == d
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

    @testset "Grids.jacobian" begin
        c = Circle([0,0], 1)
        @test Grids.jacobian(c, 0) ≈ [0,1]
        @test Grids.jacobian(c, π/2) ≈ [-1,0]
        @test Grids.jacobian(c, π) ≈ [0,-1]
        @test Grids.jacobian(c, 3π/2) ≈ [1,0]
        @test Grids.jacobian(c, π/4) ≈ [-1/√(2),1/√(2)]

        c = Circle([0,0], 2)
        @test Grids.jacobian(c, 0) ≈ [0,2]
        @test Grids.jacobian(c, π/2) ≈ [-2,0]
        @test Grids.jacobian(c, π) ≈ [0,-2]
        @test Grids.jacobian(c, 3π/2) ≈ [2,0]
        @test Grids.jacobian(c, π/4) ≈ [-√(2),√(2)]

        c = Circle([-1,1], 1)
        @test Grids.jacobian(c, 0) ≈ [0,1]
        @test Grids.jacobian(c, π/2) ≈ [-1,0]
        @test Grids.jacobian(c, π) ≈ [0,-1]
        @test Grids.jacobian(c, 3π/2) ≈ [1,0]
        @test Grids.jacobian(c, π/4) ≈ [-1/√(2),1/√(2)]

        c = Circle([-1,1], 2)
        @test Grids.jacobian(c, 0) ≈ [0,2]
        @test Grids.jacobian(c, π/2) ≈ [-2,0]
        @test Grids.jacobian(c, π) ≈ [0,-2]
        @test Grids.jacobian(c, 3π/2) ≈ [2,0]
        @test Grids.jacobian(c, π/4) ≈ [-√(2),√(2)]
    end
end

@testset "TransfiniteInterpolationSurface" begin
    @testset "Constructors" begin
        @test TransfiniteInterpolationSurface(t->[1,2], t->[2,1], t->[0,0], t->[1,1]) isa TransfiniteInterpolationSurface

        cs = polygon_edges([0,0],[1,0],[1,1],[0,1])
        @test TransfiniteInterpolationSurface(cs...) isa TransfiniteInterpolationSurface
    end

    @testset "Evaluation" begin
        a, b, c, d = [1,0],[2,1/4],[2.5,1],[-1/3,1]
        cs = polygon_edges([0,0],[1,0],[1,1],[0,1])
        ti = TransfiniteInterpolationSurface(cs...)

        @test ti(0,0) == [0,0]
        @test ti([0,0]) == [0,0]
        @test ti(1,0) == [1,0]
        @test ti([1,0]) == [1,0]
        @test ti(1,1) == [1,1]
        @test ti([1,1]) == [1,1]
        @test ti(0,1) == [0,1]
        @test ti([0,1]) == [0,1]

        @test ti(1/2, 0) == [1/2, 0]
        @test ti(1/2, 1) == [1/2, 1]
        @test ti(0,1/2) == [0,1/2]
        @test ti(1,1/2) == [1,1/2]


        a, b, c, d = [1,0],[2,1/4],[2.5,1],[-1/3,1]
        cs = polygon_edges(a,b,c,d)
        ti = TransfiniteInterpolationSurface(cs...)

        @test ti(0,0) == a
        @test ti(1,0) == b
        @test ti(1,1) == c
        @test ti(0,1) == d

        @test ti(1/2, 0) == (a+b)/2
        @test ti(1/2, 1) == (c+d)/2
        @test ti(0, 1/2) == (a+d)/2
        @test ti(1, 1/2) == (b+c)/2

        # TODO: Some test with curved edges?
    end

    @testset "check_transfiniteinterpolation" begin
        cs = polygon_edges([0,0],[1,0],[1,1],[0,1])
        ti = TransfiniteInterpolationSurface(cs...)

        @test check_transfiniteinterpolation(ti) == nothing
        @test check_transfiniteinterpolation(Bool, ti) == true

        bad_sides = [
            LineSegment([0,0],[1/2,0]),
            LineSegment([1,0],[1,1/2]),
            LineSegment([1,1],[0,1/2]),
            LineSegment([0,1],[0,1/2]),
        ]

        s1 = TransfiniteInterpolationSurface(bad_sides[1],cs[2],cs[3],cs[4])
        s2 = TransfiniteInterpolationSurface(cs[1],bad_sides[2],cs[3],cs[4])
        s3 = TransfiniteInterpolationSurface(cs[1],cs[2],bad_sides[3],cs[4])
        s4 = TransfiniteInterpolationSurface(cs[1],cs[2],cs[3],bad_sides[4])

        @test check_transfiniteinterpolation(Bool, s1) == false
        @test check_transfiniteinterpolation(Bool, s2) == false
        @test check_transfiniteinterpolation(Bool, s3) == false
        @test check_transfiniteinterpolation(Bool, s4) == false

        @test_throws Exception check_transfiniteinterpolation(s1)
        @test_throws Exception check_transfiniteinterpolation(s2)
        @test_throws Exception check_transfiniteinterpolation(s3)
        @test_throws Exception check_transfiniteinterpolation(s4)
    end
end
