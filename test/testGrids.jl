using Sbplib.Grids
using Test

@testset "Grids" begin

@testset "EquidistantGrid" begin
    @test EquidistantGrid(4,0,1) isa EquidistantGrid
    @test dimension(EquidistantGrid(4,0,1)) == 1
    @test EquidistantGrid(4,0,1) == EquidistantGrid((4,),(0,),(1,))

    g = EquidistantGrid((5,3), (0,0), (2,1))
    @test subgrid(g, 1) == EquidistantGrid(5,0,2)
    @test subgrid(g, 2) == EquidistantGrid(3,0,1)
end

end
