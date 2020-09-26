using Sbplib.Grids
using Test

@testset "Grids" begin

@testset "EquidistantGrid" begin
    @test EquidistantGrid(4,0,1) isa EquidistantGrid
    @test dimension(EquidistantGrid(4,0,1)) == 1
    @test EquidistantGrid(4,0,1) == EquidistantGrid((4,),(0,),(1,))
end

end
