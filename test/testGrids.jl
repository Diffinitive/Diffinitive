using Sbplib.Grids
using Test

@testset "Grids" begin

@testset "EquidistantGrid" begin
    @test EquidistantGrid(4,0.0,1.0) isa EquidistantGrid
    @test EquidistantGrid(4,0.0,8.0) isa EquidistantGrid
    @test dimension(EquidistantGrid(4,0.0,1.0)) == 1
    @test EquidistantGrid(4,0.0,1.0) == EquidistantGrid((4,),(0.0,),(1.0,))

    g = EquidistantGrid((5,3), (0.0,0.0), (2.0,1.0))
    @test subgrid(g, 1) == EquidistantGrid(5,0.0,2.0)
    @test subgrid(g, 2) == EquidistantGrid(3,0.0,1.0)

    g = EquidistantGrid((2,5,3), (0.0,0.0,0.0), (2.0,1.0,3.0))
    @test subgrid(g, 1) == EquidistantGrid(2,0.0,2.0)
    @test subgrid(g, 2) == EquidistantGrid(5,0.0,1.0)
    @test subgrid(g, 3) == EquidistantGrid(3,0.0,3.0)
    @test subgrid(g, 1:2) == EquidistantGrid((2,5),(0.0,0.0),(2.0,1.0))
    @test subgrid(g, 2:3) == EquidistantGrid((5,3),(0.0,0.0),(1.0,3.0))
    @test subgrid(g, [1,3]) == EquidistantGrid((2,3),(0.0,0.0),(2.0,3.0))
    @test subgrid(g, [2,1]) == EquidistantGrid((5,2),(0.0,0.0),(1.0,2.0))
end

end
