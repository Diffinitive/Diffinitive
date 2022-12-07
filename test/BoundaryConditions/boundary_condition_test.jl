using Test

using Sbplib.BoundaryConditions
using Sbplib.Grids

grid_1D = EquidistantGrid(11, 0.0, 1.0)
grid_2D = EquidistantGrid((11,15), (0.0, 0.0), (1.0,1.0))
grid_3D = EquidistantGrid((11,15,13), (0.0, 0.0, 0.0), (1.0,1.0, 1.0))
(id_l,_) = boundary_identifiers(grid_1D)
(_,_,_,id_n) = boundary_identifiers(grid_2D)
(_,_,_,_,id_b,_) = boundary_identifiers(grid_3D)

@testset "BoundaryData" begin
    
    @testset "ConstantBoundaryData" begin
        c = float(pi)
        @test ConstantBoundaryData(c) isa BoundaryData
        g_1D = discretize(ConstantBoundaryData(c),boundary_grid(grid_1D, id_l))
        g_2D = discretize(ConstantBoundaryData(c),boundary_grid(grid_2D, id_n))
        @test g_1D isa Function
        @test g_2D isa Function
        @test g_1D(0.) == fill(c)
        @test g_2D(2.) == c*ones(11)
        @test_throws MethodError g_1D(0.,0.)
        @test_throws MethodError g_2D(0.,0.)
    end

    @testset "TimeDependentBoundaryData" begin
        f(t) = 1. /(t+0.1)
        @test TimeDependentBoundaryData(f) isa BoundaryData
        g_1D = discretize(TimeDependentBoundaryData(f),boundary_grid(grid_1D, id_l))
        g_2D = discretize(TimeDependentBoundaryData(f),boundary_grid(grid_2D, id_n))
        @test g_1D isa Function
        @test g_2D isa Function
        @test g_1D(0.) == f(0.)*fill(1)
        @test g_2D(2.) == f(2.)*ones(11)
        @test_throws MethodError g_1D(0.,0.)
        @test_throws MethodError g_2D(0.,0.)
    end
    
    #TBD: Is it reasoanble to have SpaceDependentBoundaryData for 1D-grids? It would then be a constant
    #     which then may be represented by ConstantBoundaryData.
    @testset "SpaceDependentBoundaryData" begin
        f0() = 2
        f1(x) = x.^2
        f2(x,y) = x.^2 - y
        @test SpaceDependentBoundaryData(f1) isa BoundaryData
        g_1D = discretize(SpaceDependentBoundaryData(f0),boundary_grid(grid_1D, id_l))
        g_2D = discretize(SpaceDependentBoundaryData(f1),boundary_grid(grid_2D, id_n))
        g_3D = discretize(SpaceDependentBoundaryData(f2),boundary_grid(grid_3D, id_n))
        @test g_1D isa Function
        @test g_2D isa Function
        @test g_3D isa Function
        @test_broken g_1D(1.) == fill(f0()) # Does not work since evalOn for f0 returns ().
        @test g_2D(2.) ≈ f1.(range(0., 1., 11)) rtol=1e-14
        @test g_3D(0.) ≈ evalOn(boundary_grid(grid_3D, id_n),f2) rtol=1e-14
        @test_throws MethodError g_1D(0.,0.)
        @test_throws MethodError g_2D(0.,0.)
        @test_throws MethodError g_3D(0.,0.)
    end
    
    # TBD: Include tests for 1D-grids? See TBD above
    @testset "SpaceTimeDependentBoundaryData" begin
        fx1(x) = x.^2
        fx2(x,y) = x.^2 - y
        ft(t) = exp(t)
        ftx1(t,x) = ft(t)*fx1(x)
        ftx2(t,x,y) = ft(t)*fx2(x,y)
        @test SpaceTimeDependentBoundaryData(ftx1) isa BoundaryData
        g_2D = discretize(SpaceTimeDependentBoundaryData(ftx1),boundary_grid(grid_2D, id_n))
        g_3D = discretize(SpaceTimeDependentBoundaryData(ftx2),boundary_grid(grid_3D, id_b))
        @test g_2D isa Function
        @test g_3D isa Function
        @test g_2D(2.) ≈ ft(2.)*fx1.(range(0., 1., 11)) rtol=1e-14
        @test g_3D(3.14) ≈ ft(3.14)*evalOn(boundary_grid(grid_3D, id_b),fx2) rtol=1e-14
        @test_throws MethodError g_2D(0.,0.)
        @test_throws MethodError g_3D(0.,0.)
    end

    @testset "ZeroBoundaryData" begin
        @test ZeroBoundaryData() isa BoundaryData
        g_2D = discretize(ZeroBoundaryData(), boundary_grid(grid_2D, id_n))
        g_3D = discretize(ZeroBoundaryData(), boundary_grid(grid_3D, id_b))
        @test g_2D isa Function
        @test g_3D isa Function
        @test g_2D(2.) ≈ 0.0*range(0., 1., 11) rtol=1e-14
        f(x,y) = 0
        @test g_3D(3.14) ≈ 0.0*evalOn(boundary_grid(grid_3D, id_b), f) rtol=1e-14
        @test_throws MethodError g_2D(0.,0.)
        @test_throws MethodError g_3D(0.,0.)
    end
end

@testset "BoundaryCondition" begin
    g = ConstantBoundaryData(1.0)
    NeumannCondition(g,id_n) isa BoundaryCondition{ConstantBoundaryData}
    DirichletCondition(g,id_n) isa BoundaryCondition{ConstantBoundaryData}
    @test data(NeumannCondition(g,id_n)) == g
end
