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
end

@testset "eval_on" begin
    @test eval_on(ZeroDimGrid(@SVector[1.,2.]), x̄->x̄[1]+x̄[2]) isa LazyArray
    @test eval_on(ZeroDimGrid(@SVector[1.,2.]), x̄->x̄[1]+x̄[2]) == fill(3.)
    @test eval_on(ZeroDimGrid(@SVector[3.,2.]), x̄->x̄[1]+x̄[2]) == fill(5.)

    @test eval_on(EquidistantGrid(range(0,1,length=4)), x->2x) isa LazyArray
    @test eval_on(EquidistantGrid(range(0,1,length=4)), x->2x) == 2 .* range(0,1,length=4)


    g = equidistant_grid((5,3), (0.0,0.0), (2.0,1.0))


    # Splat for only one dim, controllef by type specification in function.

    @test eval_on(g, x̄ -> 0.) isa LazyArray
    @test eval_on(g, x̄ -> 0.) == fill(0., (5,3))

    @test eval_on(g, x̄ -> sin(x̄[1])*cos(x̄[2])) == map(x̄->sin(x̄[1])*cos(x̄[2]), g)

    # Vector valued function
    @test eval_on(g, x̄ -> @SVector[x̄[2], x̄[1]]) isa LazyArray{SVector{2,Float64}}
    @test eval_on(g, x̄ -> @SVector[x̄[2], x̄[1]]) == map(x̄ -> @SVector[x̄[2], x̄[1]], g)



    f(x,y) = sin(x)*cos(y)
    @test_broken eval_on(g, f) == map(p->f(p...), points(g))
end

@testset "getcomponent" begin
    @test_broken false
end
