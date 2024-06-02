using Test

using Sbplib.BoundaryConditions
using Sbplib.Grids
using Sbplib.RegionIndices

@testset "BoundaryCondition" begin
    grid_1d = equidistant_grid(0.0, 1.0, 11)
    grid_2d = equidistant_grid((0.0, 0.0), (1.0,1.0), 11, 15)
    grid_3d = equidistant_grid((0.0, 0.0, 0.0), (1.0,1.0, 1.0), 11, 15, 13)
    (id_l,_) = boundary_identifiers(grid_1d)
    (_,_,_,id_n) = boundary_identifiers(grid_2d)
    (_,_,_,_,id_b,_) = boundary_identifiers(grid_3d)

    g = 3.14
    f(x,y,z) = x^2+y^2+z^2
    @testset "Constructors" begin
        @test DirichletCondition(g,id_l) isa BoundaryCondition{Lower}
        @test DirichletCondition(g,id_n) isa BoundaryCondition{CartesianBoundary{2,Upper}}
        @test DirichletCondition(g,id_l) isa DirichletCondition{Float64,Lower}
        @test NeumannCondition(f,id_b) isa NeumannCondition{<:Function}
    end

    @testset "boundary" begin
        @test boundary(DirichletCondition(g,id_l)) == id_l
        @test boundary(NeumannCondition(f,id_b)) == id_b
    end

    @testset "boundary_data" begin
        @test boundary_data(DirichletCondition(g,id_l)) == g
        @test boundary_data(NeumannCondition(f,id_b)) == f
    end

    @testset "discretize_data" begin
        @test fill(g) ≈ discretize_data(grid_1d,DirichletCondition(g,id_l))
        @test g*ones(11,1) ≈ discretize_data(grid_2d,DirichletCondition(g,id_n))
        X = repeat(0:1/10:1, inner = (1,15))
        Y = repeat(0:1/14:1, outer = (1,11))
        @test map((x,y)->f(x,y,0), X,Y') ≈ discretize_data(grid_3d,NeumannCondition(f,id_b))
    end
end
