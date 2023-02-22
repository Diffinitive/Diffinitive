using Sbplib.Grids
using Test

@testset "TensorGrid" begin
    @test_broken false


    @testset "restrict" begin
        @test_broken restrict(g, 1:2) == nothing
        @test_broken restrict(g, 2:3) == nothing
        @test_broken restrict(g, [1,3]) == nothing
        @test_broken restrict(g, [2,1]) == nothing
    end
end
