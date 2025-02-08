using Diffinitive.Grids

@testset "MultiBlockBoundary" begin
    @test MultiBlockBoundary{1,UpperBoundary}() isa BoundaryIdentifier

    @test grid_id(MultiBlockBoundary{1,UpperBoundary}()) == 1

    @test boundary_id(MultiBlockBoundary{1,UpperBoundary}()) == UpperBoundary()

end
