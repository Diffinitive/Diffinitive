using Test

using Sbplib.BoundaryConditions
using Sbplib.Grids

@testset "BoundaryCondition" begin
    grid_2d = equidistant_grid((11,15), (0.0, 0.0), (1.0,1.0))
    grid_3d = equidistant_grid((11,15,13), (0.0, 0.0, 0.0), (1.0,1.0, 1.0))
    (_,_,_,id_n) = boundary_identifiers(grid_2d)
    (_,_,_,_,id_b,_) = boundary_identifiers(grid_3d)

    g = 3.14
    f(x,y) = x^2+y^2
    @test DirichletCondition(g,id_n) isa BoundaryCondition{Float64}
    @test NeumannCondition(f,id_n) isa BoundaryCondition{<:Function}

#    g_n = discretize_data(grid_2d,DirichletCondition(f,id_n))
#    @test g_n .≈ g*ones(1,11)
end
