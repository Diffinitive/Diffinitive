using Test
using Sbplib.Grids
using Sbplib.LazyTensors
using StaticArrays

@testset "Grid" begin
    struct DummyGrid{T,D} <: Grid{T,D} end

    @test eltype(DummyGrid{Int, 2}) == Int
    @test eltype(DummyGrid{Int, 2}()) == Int

    @test ndims(DummyGrid{Int, 2}()) == 2
    @test dims(DummyGrid{Int, 2}()) == 1:2

    @test target_manifold_dim(DummyGrid{Int, 1}()) == 1
    @test target_manifold_dim(DummyGrid{SVector{3,Float64}, 2}()) == 3

    @testset "component_type" begin
        @test component_type(DummyGrid{Int,1}()) == Int
        @test component_type(DummyGrid{Float64,1}()) == Float64
        @test component_type(DummyGrid{Rational,1}()) == Rational

        @test component_type(DummyGrid{SVector{3,Int},2}()) == Int
        @test component_type(DummyGrid{SVector{2,Float64},3}()) == Float64
        @test component_type(DummyGrid{SVector{4,Rational},4}()) == Rational
    end
end

@testset "eval_on" begin
    @test eval_on(ZeroDimGrid(@SVector[1.,2.]), x̄->x̄[1]+x̄[2]) isa LazyArray
    @test eval_on(ZeroDimGrid(@SVector[1.,2.]), x̄->x̄[1]+x̄[2]) == fill(3.)
    @test eval_on(ZeroDimGrid(@SVector[3.,2.]), x̄->x̄[1]+x̄[2]) == fill(5.)

    @test eval_on(ZeroDimGrid(1.), x̄->2x̄) isa LazyArray
    @test eval_on(ZeroDimGrid(1.), x̄->2x̄) == fill(2.)

    @test eval_on(EquidistantGrid(range(0,1,length=4)), x->2x) isa LazyArray
    @test eval_on(EquidistantGrid(range(0,1,length=4)), x->2x) == 2 .* range(0,1,length=4)


    g = equidistant_grid((5,3), (0.0,0.0), (2.0,1.0))

    @test eval_on(g, x̄ -> 0.) isa LazyArray
    @test eval_on(g, x̄ -> 0.) == fill(0., (5,3))

    @test eval_on(g, x̄ -> sin(x̄[1])*cos(x̄[2])) == map(x̄->sin(x̄[1])*cos(x̄[2]), g)

    # Vector valued function
    @test eval_on(g, x̄ -> @SVector[x̄[2], x̄[1]]) isa LazyArray{SVector{2,Float64}}
    @test eval_on(g, x̄ -> @SVector[x̄[2], x̄[1]]) == map(x̄ -> @SVector[x̄[2], x̄[1]], g)

    # Multi-argument functions
    f(x,y) = sin(x)*cos(y)
    @test eval_on(g, f) == map(x̄->f(x̄...), g)
end

@testset "_ncomponents" begin
    @test Grids._ncomponents(Int) == 1
    @test Grids._ncomponents(Float64) == 1
    @test Grids._ncomponents(Rational) == 1

    @test Grids._ncomponents(SVector{3,Int}) == 3
    @test Grids._ncomponents(SVector{2,Float64}) == 2
    @test Grids._ncomponents(SVector{4,Rational}) == 4
end

@testset "_component_type" begin
    @test Grids._component_type(Int) == Int
    @test Grids._component_type(Float64) == Float64
    @test Grids._component_type(Rational) == Rational

    @test Grids._component_type(SVector{3,Int}) == Int
    @test Grids._component_type(SVector{2,Float64}) == Float64
    @test Grids._component_type(SVector{4,Rational}) == Rational
end
