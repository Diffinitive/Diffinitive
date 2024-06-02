using Test


using Sbplib.BoundaryConditions
using Sbplib.Grids
using Sbplib.LazyTensors
using Sbplib.SbpOperators

stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order = 4)

struct MockOp end

function BoundaryConditions.sat_tensors(op::MockOp, g::Grid, bc::DirichletCondition; a = 1.)
    e = boundary_restriction(g, stencil_set, boundary(bc))
    L = a*e
    sat_op = e'
    return sat_op, L
end

function BoundaryConditions.sat_tensors(op::MockOp, g::Grid, bc::NeumannCondition)
    e = boundary_restriction(g, stencil_set, boundary(bc))
    d = normal_derivative(g, stencil_set, boundary(bc))
    L = d
    sat_op = e'
    return sat_op, L
end

@testset "sat" begin
    op = MockOp()
    @testset "1D" begin
        grid  = equidistant_grid(0., 1., 11)
        l, r = boundary_identifiers(grid)
        u = eval_on(grid, x-> 1. + 2x^2)
        dc = DirichletCondition(1.0, l)
        g_l = discretize_data(grid, dc)
        SAT_l = sat(op, grid, dc)
        @test SAT_l(u, g_l) ≈ zeros((size(grid))) atol = 1e-13
        
        nc = NeumannCondition(4.0, r)
        g_r = discretize_data(grid, nc)
        SAT_r = sat(op, grid, nc)
        @test SAT_r(u, g_r) ≈ zeros((size(grid))) atol = 1e-13
    end
    @testset "2D" begin
        grid  = equidistant_grid((0.,0.), (1.,1.), 11, 13)
        W, E, S, N = boundary_identifiers(grid)
        u = eval_on(grid, (x,y) -> x+y^2)

        dc_W = DirichletCondition(1.0, W)
        SAT_W = sat(op, grid, dc_W)
        g_W = discretize_data(grid, dc_W)
        r_W = zeros(size(grid))
        r_W[1,:] .= map(y -> (y^2-1.), range(0., 1., length=13))
        @test SAT_W(u, g_W) ≈ r_W atol = 1e-13

        dc_E = DirichletCondition(2, E)
        SAT_E = sat(op, grid, dc_E; a = 2.)
        g_E = discretize_data(grid, dc_E)
        r_E = zeros(size(grid))
        r_E[end,:] .= map(y -> (2*(1. + y^2)-2.), range(0., 1., length=13))
        @test SAT_E(u, g_E) ≈ r_E atol = 1e-13

        nc_S = NeumannCondition(.0, S)
        SAT_S = sat(op, grid, nc_S)
        g_S = discretize_data(grid, nc_S)
        @test SAT_S(u, g_S) ≈ zeros(size(grid)) atol = 1e-13

        nc_N = NeumannCondition(2.0, N)
        SAT_N = sat(op, grid, nc_N)
        g_N = discretize_data(grid, nc_N)
        @test SAT_N(u, g_N) ≈ zeros(size(grid)) atol = 1e-13
    end
end
