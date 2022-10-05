using Sbplib.Grids
using Test
using Sbplib.RegionIndices


@testset "EquidistantGrid" begin
    @test EquidistantGrid(4,0.0,1.0) isa EquidistantGrid
    @test EquidistantGrid(4,0.0,8.0) isa EquidistantGrid
    # constuctor
    @test_throws DomainError EquidistantGrid(0,0.0,1.0)
    @test_throws DomainError EquidistantGrid(1,1.0,1.0)
    @test_throws DomainError EquidistantGrid(1,1.0,-1.0)
    @test EquidistantGrid(4,0.0,1.0) == EquidistantGrid((4,),(0.0,),(1.0,))

    @testset "Base" begin
        @test eltype(EquidistantGrid(4,0.0,1.0)) == Float64
        @test eltype(EquidistantGrid((4,3),(0,0),(1,3))) == Int
        @test size(EquidistantGrid(4,0.0,1.0)) == (4,)
        @test size(EquidistantGrid((5,3), (0.0,0.0), (2.0,1.0))) == (5,3)
        @test ndims(EquidistantGrid(4,0.0,1.0)) == 1
        @test ndims(EquidistantGrid((5,3), (0.0,0.0), (2.0,1.0))) == 2
    end

    @testset "spacing" begin
        @test [spacing(EquidistantGrid(4,0.0,1.0))...] ≈ [(1. /3,)...] atol=5e-13
        @test [spacing(EquidistantGrid((5,3), (0.0,-1.0), (2.0,1.0)))...] ≈ [(0.5, 1.)...] atol=5e-13
    end

    @testset "inverse_spacing" begin
        @test [inverse_spacing(EquidistantGrid(4,0.0,1.0))...] ≈ [(3.,)...] atol=5e-13
        @test [inverse_spacing(EquidistantGrid((5,3), (0.0,-1.0), (2.0,1.0)))...] ≈ [(2, 1.)...] atol=5e-13
    end

    @testset "points" begin
        g = EquidistantGrid((5,3), (-1.0,0.0), (0.0,7.11))
        gp = points(g);
        p = [(-1.,0.)      (-1.,7.11/2)   (-1.,7.11);
            (-0.75,0.)    (-0.75,7.11/2) (-0.75,7.11);
            (-0.5,0.)     (-0.5,7.11/2)  (-0.5,7.11);
            (-0.25,0.)    (-0.25,7.11/2) (-0.25,7.11);
            (0.,0.)       (0.,7.11/2)    (0.,7.11)]
        for i ∈ eachindex(gp)
            @test [gp[i]...] ≈ [p[i]...] atol=5e-13
        end
    end

    @testset "restrict" begin
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

    @testset "boundary_identifiers" begin
        g = EquidistantGrid((2,5,3), (0.0,0.0,0.0), (2.0,1.0,3.0))
        bids = (CartesianBoundary{1,Lower}(),CartesianBoundary{1,Upper}(),
                CartesianBoundary{2,Lower}(),CartesianBoundary{2,Upper}(),
                CartesianBoundary{3,Lower}(),CartesianBoundary{3,Upper}())
        @test boundary_identifiers(g) == bids
        @inferred boundary_identifiers(g)
    end

    @testset "boundary_grid" begin
            @testset "1D" begin
                g = EquidistantGrid(5,0.0,2.0)
                (id_l, id_r) = boundary_identifiers(g)
                @test boundary_grid(g,id_l) == EquidistantGrid{Float64}()
                @test boundary_grid(g,id_r) == EquidistantGrid{Float64}()
                @test_throws DomainError boundary_grid(g,CartesianBoundary{2,Lower}())
                @test_throws DomainError boundary_grid(g,CartesianBoundary{0,Lower}())
            end
            @testset "2D" begin
                g = EquidistantGrid((5,3),(0.0,0.0),(1.0,3.0))
                (id_w, id_e, id_s, id_n) = boundary_identifiers(g)
                @test boundary_grid(g,id_w) == restrict(g,2)
                @test boundary_grid(g,id_e) == restrict(g,2)
                @test boundary_grid(g,id_s) == restrict(g,1)
                @test boundary_grid(g,id_n) == restrict(g,1)
                @test_throws DomainError boundary_grid(g,CartesianBoundary{4,Lower}())
            end
            @testset "3D" begin
                g = EquidistantGrid((2,5,3), (0.0,0.0,0.0), (2.0,1.0,3.0))
                (id_w, id_e,
                 id_s, id_n,
                 id_t, id_b) = boundary_identifiers(g)
                @test boundary_grid(g,id_w) == restrict(g,[2,3])
                @test boundary_grid(g,id_e) == restrict(g,[2,3])
                @test boundary_grid(g,id_s) == restrict(g,[1,3])
                @test boundary_grid(g,id_n) == restrict(g,[1,3])
                @test boundary_grid(g,id_t) == restrict(g,[1,2])
                @test boundary_grid(g,id_b) == restrict(g,[1,2])
                @test_throws DomainError boundary_grid(g,CartesianBoundary{4,Lower}())
            end
    end

    @testset "refine" begin
        @test refine(EquidistantGrid{Float64}(), 1) == EquidistantGrid{Float64}()
        @test refine(EquidistantGrid{Float64}(), 2) == EquidistantGrid{Float64}()

        g = EquidistantGrid((10,5),(0.,1.),(2.,3.))
        @test refine(g, 1) == g
        @test refine(g, 2) == EquidistantGrid((19,9),(0.,1.),(2.,3.))
        @test refine(g, 3) == EquidistantGrid((28,13),(0.,1.),(2.,3.))
    end

    @testset "coarsen" begin
        @test coarsen(EquidistantGrid{Float64}(), 1) == EquidistantGrid{Float64}()
        @test coarsen(EquidistantGrid{Float64}(), 2) == EquidistantGrid{Float64}()

        g = EquidistantGrid((7,13),(0.,1.),(2.,3.))
        @test coarsen(g, 1) == g
        @test coarsen(g, 2) == EquidistantGrid((4,7),(0.,1.),(2.,3.))
        @test coarsen(g, 3) == EquidistantGrid((3,5),(0.,1.),(2.,3.))

        @test_throws DomainError(4, "Size minus 1 must be divisible by the ratio.") coarsen(g, 4) == EquidistantGrid((3,5),(0.,1.),(2.,3.))
    end
end
