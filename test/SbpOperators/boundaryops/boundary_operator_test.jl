using Test

using Sbplib.LazyTensors
using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.RegionIndices
import Sbplib.SbpOperators.Stencil
import Sbplib.SbpOperators.BoundaryOperator

# TODO: What should happen to all the commented tests? Deleted? Replicated for user code?

@testset "BoundaryOperator" begin
    closure_stencil = Stencil(2.,1.,3.; center = 1)
    g_1D = EquidistantGrid(11, 0.0, 1.0)
    g_2D = EquidistantGrid((11,15), (0.0, 0.0), (1.0,1.0))

    @testset "Constructors" begin
        @testset "1D" begin # TODO: Remove these testsets
            op_l = BoundaryOperator{Lower}(closure_stencil,size(g_1D)[1])
            @test op_l == BoundaryOperator(g_1D,closure_stencil,Lower())
            @test op_l isa LazyTensor{T,0,1} where T

            op_r = BoundaryOperator{Upper}(closure_stencil,size(g_1D)[1]) # TBD: Is this constructor really needed? looks weird!
            @test op_r == BoundaryOperator(g_1D,closure_stencil,Upper())
            @test op_r isa LazyTensor{T,0,1} where T
        end

        # @testset "2D" begin
        #     e_w = boundary_operator(g_2D,closure_stencil,CartesianBoundary{1,Upper}())
        #     @test e_w isa InflatedTensor
        #     @test e_w isa LazyTensor{T,1,2} where T
        # end
    end

    op_l = BoundaryOperator(g_1D, closure_stencil, Lower())
    op_r = BoundaryOperator(g_1D, closure_stencil, Upper())
    # op_w, op_e, op_s, op_n = boundary_operator.(Ref(g_2D), Ref(closure_stencil), boundary_identifiers(g_2D))

    @testset "Sizes" begin
        @testset "1D" begin
            @test domain_size(op_l) == (11,)
            @test domain_size(op_r) == (11,)

            @test range_size(op_l) == ()
            @test range_size(op_r) == ()
        end

        # @testset "2D" begin
        #     @test domain_size(op_w) == (11,15)
        #     @test domain_size(op_e) == (11,15)
        #     @test domain_size(op_s) == (11,15)
        #     @test domain_size(op_n) == (11,15)

        #     @test range_size(op_w) == (15,)
        #     @test range_size(op_e) == (15,)
        #     @test range_size(op_s) == (11,)
        #     @test range_size(op_n) == (11,)
        # end
    end

    @testset "Application" begin
        @testset "1D" begin
            v = evalOn(g_1D,x->1+x^2)
            u = fill(3.124)
            @test (op_l*v)[] == 2*v[1] + v[2] + 3*v[3]
            @test (op_r*v)[] == 2*v[end] + v[end-1] + 3*v[end-2]
            @test (op_r*v)[1] == 2*v[end] + v[end-1] + 3*v[end-2]
            @test op_l'*u == [2*u[]; u[]; 3*u[]; zeros(8)]
            @test op_r'*u == [zeros(8); 3*u[]; u[]; 2*u[]]

            v = evalOn(g_1D, x->1. +x*im)
            @test (op_l*v)[] isa ComplexF64

            u = fill(1. +im)
            @test (op_l'*u)[1] isa ComplexF64
            @test (op_l'*u)[5] isa ComplexF64
            @test (op_l'*u)[11] isa ComplexF64
        end

       #  @testset "2D" begin
       #      v = rand(size(g_2D)...)
       #      u = fill(3.124)
       #      @test op_w*v ≈ 2*v[1,:] + v[2,:] + 3*v[3,:] rtol = 1e-14
       #      @test op_e*v ≈ 2*v[end,:] + v[end-1,:] + 3*v[end-2,:] rtol = 1e-14
       #      @test op_s*v ≈ 2*v[:,1] + v[:,2] + 3*v[:,3] rtol = 1e-14
       #      @test op_n*v ≈ 2*v[:,end] + v[:,end-1] + 3*v[:,end-2] rtol = 1e-14


       #      g_x = rand(size(g_2D)[1])
       #      g_y = rand(size(g_2D)[2])

       #      G_w = zeros(Float64, size(g_2D)...)
       #      G_w[1,:] = 2*g_y
       #      G_w[2,:] = g_y
       #      G_w[3,:] = 3*g_y

       #      G_e = zeros(Float64, size(g_2D)...)
       #      G_e[end,:] = 2*g_y
       #      G_e[end-1,:] = g_y
       #      G_e[end-2,:] = 3*g_y

       #      G_s = zeros(Float64, size(g_2D)...)
       #      G_s[:,1] = 2*g_x
       #      G_s[:,2] = g_x
       #      G_s[:,3] = 3*g_x

       #      G_n = zeros(Float64, size(g_2D)...)
       #      G_n[:,end] = 2*g_x
       #      G_n[:,end-1] = g_x
       #      G_n[:,end-2] = 3*g_x

       #      @test op_w'*g_y == G_w
       #      @test op_e'*g_y == G_e
       #      @test op_s'*g_x == G_s
       #      @test op_n'*g_x == G_n
       # end

       @testset "Regions" begin
            u = fill(3.124)
            @test (op_l'*u)[Index(1,Lower)] == 2*u[]
            @test (op_l'*u)[Index(2,Lower)] == u[]
            @test (op_l'*u)[Index(6,Interior)] == 0
            @test (op_l'*u)[Index(10,Upper)] == 0
            @test (op_l'*u)[Index(11,Upper)] == 0

            @test (op_r'*u)[Index(1,Lower)] == 0
            @test (op_r'*u)[Index(2,Lower)] == 0
            @test (op_r'*u)[Index(6,Interior)] == 0
            @test (op_r'*u)[Index(10,Upper)] == u[]
            @test (op_r'*u)[Index(11,Upper)] == 2*u[]
       end
    end

    @testset "Inferred" begin
        v = ones(Float64, 11)
        u = fill(1.)

        @inferred apply(op_l, v)
        @inferred apply(op_r, v)

        @inferred apply_transpose(op_l, u, 4)
        @inferred apply_transpose(op_l, u, Index(1,Lower))
        @inferred apply_transpose(op_l, u, Index(2,Lower))
        @inferred apply_transpose(op_l, u, Index(6,Interior))
        @inferred apply_transpose(op_l, u, Index(10,Upper))
        @inferred apply_transpose(op_l, u, Index(11,Upper))

        @inferred apply_transpose(op_r, u, 4)
        @inferred apply_transpose(op_r, u, Index(1,Lower))
        @inferred apply_transpose(op_r, u, Index(2,Lower))
        @inferred apply_transpose(op_r, u, Index(6,Interior))
        @inferred apply_transpose(op_r, u, Index(10,Upper))
        @inferred apply_transpose(op_r, u, Index(11,Upper))
    end
end
