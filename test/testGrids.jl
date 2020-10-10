using Sbplib.Grids
using Test

@testset "Grids" begin

@testset "EquidistantGrid" begin
    @test EquidistantGrid(4,0.0,1.0) isa EquidistantGrid
    @test EquidistantGrid(4,0.0,8.0) isa EquidistantGrid
    # constuctor
    @test_throws DomainError EquidistantGrid(0,0.0,1.0)
    @test_throws DomainError EquidistantGrid(1,1.0,1.0)
    @test_throws DomainError EquidistantGrid(1,1.0,-1.0) # TODO: Remove if we allow side lengths to be negative.
    @test EquidistantGrid(4,0.0,1.0) == EquidistantGrid((4,),(0.0,),(1.0,))

    # size
    @test size(EquidistantGrid(4,0.0,1.0)) == (4,)
    @test size(EquidistantGrid((5,3), (0.0,0.0), (2.0,1.0))) == (5,3)

    # dimension
    @test dimension(EquidistantGrid(4,0.0,1.0)) == 1
    @test dimension(EquidistantGrid((5,3), (0.0,0.0), (2.0,1.0))) == 2

    # spacing
    @test [spacing(EquidistantGrid(4,0.0,1.0))...] ≈ [(1. /3,)...] atol=5e-13
    @test [spacing(EquidistantGrid((5,3), (0.0,-1.0), (2.0,1.0)))...] ≈ [(0.5, 1.)...] atol=5e-13
    # TODO: Include below if we allow side lengths to be negative.
    #@test [spacing(EquidistantGrid((5,3), (0.0,1.0), (-1.0,-2.0)))...] ≈ [(0.25, 1.5)...] atol=5e-13

    # inverse_spacing
    @test [inverse_spacing(EquidistantGrid(4,0.0,1.0))...] ≈ [(3.,)...] atol=5e-13
    @test [inverse_spacing(EquidistantGrid((5,3), (0.0,-1.0), (2.0,1.0)))...] ≈ [(2, 1.)...] atol=5e-13
    # TODO: Include below if we allow side lengths to be negative.
    #@test [inverse_spacing(EquidistantGrid((5,3), (0.0,1.0), (-1.0,-2.0)))...] ≈ [(4., 2. /3)...] atol=5e-13

    # points
    g = EquidistantGrid((5,3), (-1.0,0.0), (0.0,7.11))
    gp = points(g);
    p = [(-1.,0.)      (-1.,7.11/2)   (-1.,7.11);
         (-0.75,0.)    (-0.75,7.11/2) (-0.75,7.11);
         (-0.5,0.)     (-0.5,7.11/2)  (-0.5,7.11);
         (-0.25,0.)    (-0.25,7.11/2) (-0.25,7.11);
         (0.,0.)       (0.,7.11/2)    (0.,7.11)]
    approxequal = true;
    for i ∈ eachindex(gp)
        approxequal=approxequal*all(isapprox.(gp[i],p[i], atol=5e-13));
    end
    @test approxequal == true


    # restrict
    g = EquidistantGrid((5,3), (0.0,0.0), (2.0,1.0))
    @test restrict(g, 1) == EquidistantGrid(5,0.0,2.0)
    @test restrict(g, 2) == EquidistantGrid(3,0.0,1.0)

    g = EquidistantGrid((2,5,3), (0.0,0.0,0.0), (2.0,1.0,3.0))
    @test restrict(g, 1) == EquidistantGrid(2,0.0,2.0)
    @test restrict(g, 2) == EquidistantGrid(5,0.0,1.0)
    @test restrict(g, 3) == EquidistantGrid(3,0.0,3.0)
    @test restrict(g, 1:2) == EquidistantGrid((2,5),(0.0,0.0),(2.0,1.0))
    @test restrict(g, 2:3) == EquidistantGrid((5,3),(0.0,0.0),(1.0,3.0))
    @test restrict(g, [1,3]) == EquidistantGrid((2,3),(0.0,0.0),(2.0,3.0))
    @test restrict(g, [2,1]) == EquidistantGrid((5,2),(0.0,0.0),(1.0,2.0))
end

end
