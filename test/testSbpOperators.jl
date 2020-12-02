using Test
using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.RegionIndices
using Sbplib.LazyTensors
using LinearAlgebra

@testset "SbpOperators" begin

@testset "Stencil" begin
    s = SbpOperators.Stencil((-2,2), (1.,2.,2.,3.,4.))
    @test s isa SbpOperators.Stencil{Float64, 5}

    @test eltype(s) == Float64
    @test SbpOperators.scale(s, 2) == SbpOperators.Stencil((-2,2), (2.,4.,4.,6.,8.))

    @test SbpOperators.Stencil((1,2,3,4), center=1) == SbpOperators.Stencil((0, 3),(1,2,3,4))
    @test SbpOperators.Stencil((1,2,3,4), center=2) == SbpOperators.Stencil((-1, 2),(1,2,3,4))
    @test SbpOperators.Stencil((1,2,3,4), center=4) == SbpOperators.Stencil((-3, 0),(1,2,3,4))
end

# @testset "apply_quadrature" begin
#     op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
#     h = 0.5
#
#     @test apply_quadrature(op, h, 1.0, 10, 100) == h
#
#     N = 10
#     qc = op.quadratureClosure
#     q = h.*(qc..., ones(N-2*closuresize(op))..., reverse(qc)...)
#     @assert length(q) == N
#
#     for i ∈ 1:N
#         @test apply_quadrature(op, h, 1.0, i, N) == q[i]
#     end
#
#     v = [2.,3.,2.,4.,5.,4.,3.,4.,5.,4.5]
#     for i ∈ 1:N
#         @test apply_quadrature(op, h, v[i], i, N) == q[i]*v[i]
#     end
# end

@testset "SecondDerivative" begin
    op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
    L = 3.5
    g = EquidistantGrid(101, 0.0, L)
    Dₓₓ = SecondDerivative(g,op.innerStencil,op.closureStencils)

    f0(x) = 1.
    f1(x) = x
    f2(x) = 1/2*x^2
    f3(x) = 1/6*x^3
    f4(x) = 1/24*x^4
    f5(x) = sin(x)
    f5ₓₓ(x) = -f5(x)

    v0 = evalOn(g,f0)
    v1 = evalOn(g,f1)
    v2 = evalOn(g,f2)
    v3 = evalOn(g,f3)
    v4 = evalOn(g,f4)
    v5 = evalOn(g,f5)

    @test Dₓₓ isa TensorMapping{T,1,1} where T
    @test Dₓₓ' isa TensorMapping{T,1,1} where T

    # 4th order interior stencil, 2nd order boundary stencil,
    # implies that L*v should be exact for v - monomial up to order 3.
    # Exact differentiation is measured point-wise. For other grid functions
    # the error is measured in the l2-norm.
    @test norm(Dₓₓ*v0) ≈ 0.0 atol=5e-10
    @test norm(Dₓₓ*v1) ≈ 0.0 atol=5e-10
    @test Dₓₓ*v2 ≈ v0 atol=5e-11
    @test Dₓₓ*v3 ≈ v1 atol=5e-11

    h = spacing(g)[1];
    l2(v) = sqrt(h*sum(v.^2))
    @test Dₓₓ*v4 ≈ v2  atol=5e-4 norm=l2
    @test Dₓₓ*v5 ≈ -v5 atol=5e-4 norm=l2
end


@testset "Laplace2D" begin
    op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
    Lx = 1.5
    Ly = 3.2
    g = EquidistantGrid((102,131), (0.0, 0.0), (Lx,Ly))
    L = Laplace(g, op.innerStencil, op.closureStencils)


    f0(x,y) = 2.
    f1(x,y) = x+y
    f2(x,y) = 1/2*x^2 + 1/2*y^2
    f3(x,y) = 1/6*x^3 + 1/6*y^3
    f4(x,y) = 1/24*x^4 + 1/24*y^4
    f5(x,y) = sin(x) + cos(y)
    f5ₓₓ(x,y) = -f5(x,y)

    v0 = evalOn(g,f0)
    v1 = evalOn(g,f1)
    v2 = evalOn(g,f2)
    v3 = evalOn(g,f3)
    v4 = evalOn(g,f4)
    v5 = evalOn(g,f5)
    v5ₓₓ = evalOn(g,f5ₓₓ)

    @test L isa TensorMapping{T,2,2} where T
    @test L' isa TensorMapping{T,2,2} where T

    # 4th order interior stencil, 2nd order boundary stencil,
    # implies that L*v should be exact for v - monomial up to order 3.
    # Exact differentiation is measured point-wise. For other grid functions
    # the error is measured in the H-norm.
    @test norm(L*v0) ≈ 0 atol=5e-10
    @test norm(L*v1) ≈ 0 atol=5e-10
    @test L*v2 ≈ v0 # Seems to be more accurate
    @test L*v3 ≈ v1 atol=5e-10

    h = spacing(g)
    l2(v) = sqrt(prod(h)*sum(v.^2))
    @test L*v4 ≈ v2   atol=5e-4 norm=l2
    @test L*v5 ≈ v5ₓₓ atol=5e-4 norm=l2
end

@testset "DiagonalInnerProduct" begin
    op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
    L = 2.3
    g = EquidistantGrid(77, 0.0, L)
    H = DiagonalInnerProduct(g,op.quadratureClosure)
    v = ones(Float64, size(g))

    @test H isa TensorMapping{T,1,1} where T
    @test H' isa TensorMapping{T,1,1} where T
    @test sum(H*v) ≈ L
    @test H*v == H'*v
end

@testset "Quadrature" begin
    op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
    Lx = 2.3
    Ly = 5.2
    g = EquidistantGrid((77,66), (0.0, 0.0), (Lx,Ly))

    Q = Quadrature(g, op.quadratureClosure)

    @test Q isa TensorMapping{T,2,2} where T
    @test Q' isa TensorMapping{T,2,2} where T

    v = ones(Float64, size(g))
    @test sum(Q*v) ≈ Lx*Ly

    v = 2*ones(Float64, size(g))
    @test_broken sum(Q*v) ≈ 2*Lx*Ly

    @test Q*v == Q'*v
end

@testset "InverseDiagonalInnerProduct" begin
    op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
    L = 2.3
    g = EquidistantGrid(77, 0.0, L)
    H = DiagonalInnerProduct(g, op.quadratureClosure)
    Hi = InverseDiagonalInnerProduct(g,op.quadratureClosure)
    v = evalOn(g, x->sin(x))

    @test Hi isa TensorMapping{T,1,1} where T
    @test Hi' isa TensorMapping{T,1,1} where T
    @test Hi*H*v ≈ v
    @test Hi*v == Hi'*v
end

@testset "InverseQuadrature" begin
    op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
    Lx = 7.3
    Ly = 8.2
    g = EquidistantGrid((77,66), (0.0, 0.0), (Lx,Ly))

    Q = Quadrature(g, op.quadratureClosure)
    Qinv = InverseQuadrature(g, op.quadratureClosure)
    v = evalOn(g, (x,y)-> x^2 + (y-1)^2 + x*y)

    @test Qinv isa TensorMapping{T,2,2} where T
    @test Qinv' isa TensorMapping{T,2,2} where T
    @test_broken Qinv*(Q*v) ≈ v
    @test Qinv*v == Qinv'*v
end

@testset "BoundaryRestrictrion" begin
    op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
    g_1D = EquidistantGrid(11, 0.0, 1.0)
    g_2D = EquidistantGrid((11,15), (0.0, 0.0), (1.0,1.0))

    @testset "Constructors" begin
        @testset "1D" begin
            e_l = BoundaryRestriction{Lower}(op.eClosure,size(g_1D)[1])
            @test e_l == BoundaryRestriction(g_1D,op.eClosure,Lower())
            @test e_l == boundary_restriction(g_1D,op.eClosure,CartesianBoundary{1,Lower}())
            @test e_l isa TensorMapping{T,0,1} where T

            e_r = BoundaryRestriction{Upper}(op.eClosure,size(g_1D)[1])
            @test e_r == BoundaryRestriction(g_1D,op.eClosure,Upper())
            @test e_r == boundary_restriction(g_1D,op.eClosure,CartesianBoundary{1,Upper}())
            @test e_r isa TensorMapping{T,0,1} where T
        end

        @testset "2D" begin
            e_w = boundary_restriction(g_2D,op.eClosure,CartesianBoundary{1,Upper}())
            @test e_w isa InflatedTensorMapping
            @test e_w isa TensorMapping{T,1,2} where T
        end
    end

    e_l = boundary_restriction(g_1D, op.eClosure, CartesianBoundary{1,Lower}())
    e_r = boundary_restriction(g_1D, op.eClosure, CartesianBoundary{1,Upper}())

    e_w = boundary_restriction(g_2D, op.eClosure, CartesianBoundary{1,Lower}())
    e_e = boundary_restriction(g_2D, op.eClosure, CartesianBoundary{1,Upper}())
    e_s = boundary_restriction(g_2D, op.eClosure, CartesianBoundary{2,Lower}())
    e_n = boundary_restriction(g_2D, op.eClosure, CartesianBoundary{2,Upper}())

    @testset "Sizes" begin
        @testset "1D" begin
            @test domain_size(e_l) == (11,)
            @test domain_size(e_r) == (11,)

            @test range_size(e_l) == ()
            @test range_size(e_r) == ()
        end

        @testset "2D" begin
            @test domain_size(e_w) == (11,15)
            @test domain_size(e_e) == (11,15)
            @test domain_size(e_s) == (11,15)
            @test domain_size(e_n) == (11,15)

            @test range_size(e_w) == (15,)
            @test range_size(e_e) == (15,)
            @test range_size(e_s) == (11,)
            @test range_size(e_n) == (11,)
        end
    end


    @testset "Application" begin
        @testset "1D" begin
            v = evalOn(g_1D,x->1+x^2)
            u = fill(3.124)
            @test (e_l*v)[] == v[1]
            @test (e_r*v)[] == v[end]
            @test (e_r*v)[1] == v[end]
            @test e_l'*u == [u[]; zeros(10)]
            @test e_r'*u == [zeros(10); u[]]
        end

        @testset "2D" begin
            v = rand(11, 15)
            u = fill(3.124)

            @test e_w*v == v[1,:]
            @test e_e*v == v[end,:]
            @test e_s*v == v[:,1]
            @test e_n*v == v[:,end]


           g_x = rand(11)
           g_y = rand(15)

           G_w = zeros(Float64, (11,15))
           G_w[1,:] = g_y

           G_e = zeros(Float64, (11,15))
           G_e[end,:] = g_y

           G_s = zeros(Float64, (11,15))
           G_s[:,1] = g_x

           G_n = zeros(Float64, (11,15))
           G_n[:,end] = g_x

           @test e_w'*g_y == G_w
           @test e_e'*g_y == G_e
           @test e_s'*g_x == G_s
           @test e_n'*g_x == G_n
       end

       @testset "Regions" begin
           u = fill(3.124)
           @test (e_l'*u)[Index(1,Lower)] == 3.124
           @test (e_l'*u)[Index(2,Lower)] == 0
           @test (e_l'*u)[Index(6,Interior)] == 0
           @test (e_l'*u)[Index(10,Upper)] == 0
           @test (e_l'*u)[Index(11,Upper)] == 0

           @test (e_r'*u)[Index(1,Lower)] == 0
           @test (e_r'*u)[Index(2,Lower)] == 0
           @test (e_r'*u)[Index(6,Interior)] == 0
           @test (e_r'*u)[Index(10,Upper)] == 0
           @test (e_r'*u)[Index(11,Upper)] == 3.124
       end
    end

    @testset "Inferred" begin
        v = ones(Float64, 11)
        u = fill(1.)

        @inferred apply(e_l, v)
        @inferred apply(e_r, v)

        @inferred apply_transpose(e_l, u, 4)
        @inferred apply_transpose(e_l, u, Index(1,Lower))
        @inferred apply_transpose(e_l, u, Index(2,Lower))
        @inferred apply_transpose(e_l, u, Index(6,Interior))
        @inferred apply_transpose(e_l, u, Index(10,Upper))
        @inferred apply_transpose(e_l, u, Index(11,Upper))

        @inferred apply_transpose(e_r, u, 4)
        @inferred apply_transpose(e_r, u, Index(1,Lower))
        @inferred apply_transpose(e_r, u, Index(2,Lower))
        @inferred apply_transpose(e_r, u, Index(6,Interior))
        @inferred apply_transpose(e_r, u, Index(10,Upper))
        @inferred apply_transpose(e_r, u, Index(11,Upper))
    end

end
#
# @testset "NormalDerivative" begin
#     op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
#     g = EquidistantGrid((5,6), (0.0, 0.0), (4.0,5.0))
#
#     d_w = NormalDerivative(op, g, CartesianBoundary{1,Lower}())
#     d_e = NormalDerivative(op, g, CartesianBoundary{1,Upper}())
#     d_s = NormalDerivative(op, g, CartesianBoundary{2,Lower}())
#     d_n = NormalDerivative(op, g, CartesianBoundary{2,Upper}())
#
#
#     v = evalOn(g, (x,y)-> x^2 + (y-1)^2 + x*y)
#     v∂x = evalOn(g, (x,y)-> 2*x + y)
#     v∂y = evalOn(g, (x,y)-> 2*(y-1) + x)
#
#     @test d_w  isa TensorMapping{T,2,1} where T
#     @test d_w' isa TensorMapping{T,1,2} where T
#
#     @test domain_size(d_w, (3,2)) == (2,)
#     @test domain_size(d_e, (3,2)) == (2,)
#     @test domain_size(d_s, (3,2)) == (3,)
#     @test domain_size(d_n, (3,2)) == (3,)
#
#     @test size(d_w'*v) == (6,)
#     @test size(d_e'*v) == (6,)
#     @test size(d_s'*v) == (5,)
#     @test size(d_n'*v) == (5,)
#
#     @test d_w'*v .≈ v∂x[1,:]
#     @test d_e'*v .≈ v∂x[5,:]
#     @test d_s'*v .≈ v∂y[:,1]
#     @test d_n'*v .≈ v∂y[:,6]
#
#
#     d_x_l = zeros(Float64, 5)
#     d_x_u = zeros(Float64, 5)
#     for i ∈ eachindex(d_x_l)
#         d_x_l[i] = op.dClosure[i-1]
#         d_x_u[i] = -op.dClosure[length(d_x_u)-i]
#     end
#
#     d_y_l = zeros(Float64, 6)
#     d_y_u = zeros(Float64, 6)
#     for i ∈ eachindex(d_y_l)
#         d_y_l[i] = op.dClosure[i-1]
#         d_y_u[i] = -op.dClosure[length(d_y_u)-i]
#     end
#
#     function prod_matrix(x,y)
#         G = zeros(Float64, length(x), length(y))
#         for I ∈ CartesianIndices(G)
#             G[I] = x[I[1]]*y[I[2]]
#         end
#
#         return G
#     end
#
#     g_x = [1,2,3,4.0,5]
#     g_y = [5,4,3,2,1.0,11]
#
#     G_w = prod_matrix(d_x_l, g_y)
#     G_e = prod_matrix(d_x_u, g_y)
#     G_s = prod_matrix(g_x, d_y_l)
#     G_n = prod_matrix(g_x, d_y_u)
#
#
#     @test size(d_w*g_y) == (UnknownDim,6)
#     @test size(d_e*g_y) == (UnknownDim,6)
#     @test size(d_s*g_x) == (5,UnknownDim)
#     @test size(d_n*g_x) == (5,UnknownDim)
#
#     # These tests should be moved to where they are possible (i.e we know what the grid should be)
#     @test_broken d_w*g_y .≈ G_w
#     @test_broken d_e*g_y .≈ G_e
#     @test_broken d_s*g_x .≈ G_s
#     @test_broken d_n*g_x .≈ G_n
# end
#
# @testset "BoundaryQuadrature" begin
#     op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
#     g = EquidistantGrid((10,11), (0.0, 0.0), (1.0,1.0))
#
#     H_w = BoundaryQuadrature(op, g, CartesianBoundary{1,Lower}())
#     H_e = BoundaryQuadrature(op, g, CartesianBoundary{1,Upper}())
#     H_s = BoundaryQuadrature(op, g, CartesianBoundary{2,Lower}())
#     H_n = BoundaryQuadrature(op, g, CartesianBoundary{2,Upper}())
#
#     v = evalOn(g, (x,y)-> x^2 + (y-1)^2 + x*y)
#
#     function get_quadrature(N)
#         qc = op.quadratureClosure
#         q = (qc..., ones(N-2*closuresize(op))..., reverse(qc)...)
#         @assert length(q) == N
#         return q
#     end
#
#     v_w = v[1,:]
#     v_e = v[10,:]
#     v_s = v[:,1]
#     v_n = v[:,11]
#
#     q_x = spacing(g)[1].*get_quadrature(10)
#     q_y = spacing(g)[2].*get_quadrature(11)
#
#     @test H_w isa TensorOperator{T,1} where T
#
#     @test domain_size(H_w, (3,)) == (3,)
#     @test domain_size(H_n, (3,)) == (3,)
#
#     @test range_size(H_w, (3,)) == (3,)
#     @test range_size(H_n, (3,)) == (3,)
#
#     @test size(H_w*v_w) == (11,)
#     @test size(H_e*v_e) == (11,)
#     @test size(H_s*v_s) == (10,)
#     @test size(H_n*v_n) == (10,)
#
#     @test H_w*v_w .≈ q_y.*v_w
#     @test H_e*v_e .≈ q_y.*v_e
#     @test H_s*v_s .≈ q_x.*v_s
#     @test H_n*v_n .≈ q_x.*v_n
#
#     @test H_w'*v_w == H_w'*v_w
#     @test H_e'*v_e == H_e'*v_e
#     @test H_s'*v_s == H_s'*v_s
#     @test H_n'*v_n == H_n'*v_n
# end

end
