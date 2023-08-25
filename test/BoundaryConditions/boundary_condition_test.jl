using Test

using Sbplib.BoundaryConditions
using Sbplib.Grids

@testset "BoundaryCondition" begin
    grid_2d = equidistant_grid((11,15), (0.0, 0.0), (1.0,1.0))
    grid_3d = equidistant_grid((11,15,13), (0.0, 0.0, 0.0), (1.0,1.0, 1.0))
    (_,_,_,id_n) = boundary_identifiers(grid_2d)
    (_,_,_,_,id_b,_) = boundary_identifiers(grid_3d)

    g = 3.14
    f(x,y,z) = x^2+y^2+z^2
    @test DirichletCondition(g,id_n) isa BoundaryCondition{Float64}
    @test NeumannCondition(f,id_b) isa BoundaryCondition{<:Function}

    @test g*ones(11,1) ≈ discretize_data(grid_2d,DirichletCondition(g,id_n))
    X = repeat(0:1/10:1, inner = (1,15))
    Y = repeat(0:1/14:1, outer = (1,11))
    @test map((x,y)->f(x,y,0), X,Y') ≈ discretize_data(grid_3d,NeumannCondition(f,id_b))
end
