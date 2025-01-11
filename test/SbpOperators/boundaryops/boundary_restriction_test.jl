using Test

using Diffinitive.SbpOperators
using Diffinitive.Grids
using Diffinitive.LazyTensors
using Diffinitive.RegionIndices
using Diffinitive.SbpOperators: BoundaryOperator, Stencil

@testset "boundary_restriction" begin
	stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order = 4)
	e_closure = parse_stencil(stencil_set["e"]["closure"])
    g_1D = equidistant_grid(0.0, 1.0, 11)
    g_2D = equidistant_grid((0.0, 0.0), (1.0,1.0), 11, 15)

    @testset "boundary_restriction" begin
        @testset "1D" begin
            e_l = boundary_restriction(g_1D,stencil_set,LowerBoundary())
            @test e_l == BoundaryOperator(g_1D,Stencil{Float64}(e_closure),LowerBoundary())
            @test e_l isa BoundaryOperator{T,LowerBoundary} where T
            @test e_l isa LazyTensor{T,0,1} where T

            e_r = boundary_restriction(g_1D,stencil_set,UpperBoundary())
            @test e_r == BoundaryOperator(g_1D,Stencil{Float64}(e_closure),UpperBoundary())
            @test e_r isa BoundaryOperator{T,UpperBoundary} where T
            @test e_r isa LazyTensor{T,0,1} where T
        end

        @testset "2D" begin
            e_w = boundary_restriction(g_2D,stencil_set,CartesianBoundary{1,UpperBoundary}())
            @test e_w isa InflatedTensor
            @test e_w isa LazyTensor{T,1,2} where T
        end
    end

    @testset "Application" begin
        @testset "1D" begin
            e_l, e_r = boundary_restriction.(Ref(g_1D), Ref(stencil_set), boundary_identifiers(g_1D))
            v = eval_on(g_1D,x->1+x^2)
            u = fill(3.124)

            @test (e_l*v)[] == v[1]
            @test (e_r*v)[] == v[end]
            @test (e_r*v)[1] == v[end]
        end

        @testset "2D" begin
            e_w, e_e, e_s, e_n = boundary_restriction.(Ref(g_2D), Ref(stencil_set), boundary_identifiers(g_2D))
            v = rand(11, 15)
            u = fill(3.124)

            @test e_w*v == v[1,:]
            @test e_e*v == v[end,:]
            @test e_s*v == v[:,1]
            @test e_n*v == v[:,end]
       end
    end
end
