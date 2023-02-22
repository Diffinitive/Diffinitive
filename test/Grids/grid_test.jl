using Sbplib.Grids
using Test

@testset "Grid" begin
    @test_broken false
end

@testset "eval_on" begin
    g = equidistant_grid((5,3), (0.0,0.0), (2.0,1.0))

    @test_broken eval_on(g, (x,y) -> 0.) isa LazyArray
    @test_broken eval_on(g, (x,y) -> 0.) == fill(0., (5,3))

    f(x,y) = sin(x)*cos(y)
    @test_broken eval_on(g, f) == map(p->f(p...), points(g))
end

@testset "getcomponent" begin
    @test_broken false
end
