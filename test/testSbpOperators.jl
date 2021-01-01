using Test
using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.RegionIndices
using Sbplib.LazyTensors
using LinearAlgebra
using TOML

import Sbplib.SbpOperators.Stencil

@testset "SbpOperators" begin

@testset "Stencil" begin
    s = Stencil((-2,2), (1.,2.,2.,3.,4.))
    @test s isa Stencil{Float64, 5}

    @test eltype(s) == Float64
    @test SbpOperators.scale(s, 2) == Stencil((-2,2), (2.,4.,4.,6.,8.))

    @test Stencil((1,2,3,4), center=1) == Stencil((0, 3),(1,2,3,4))
    @test Stencil((1,2,3,4), center=2) == Stencil((-1, 2),(1,2,3,4))
    @test Stencil((1,2,3,4), center=4) == Stencil((-3, 0),(1,2,3,4))
end

@testset "parse_rational" begin
    @test SbpOperators.parse_rational("1") isa Rational
    @test SbpOperators.parse_rational("1") == 1//1
    @test SbpOperators.parse_rational("1/2") isa Rational
    @test SbpOperators.parse_rational("1/2") == 1//2
    @test SbpOperators.parse_rational("37/13") isa Rational
    @test SbpOperators.parse_rational("37/13") == 37//13
end

@testset "readoperator" begin
    toml_str = """
        [meta]
        type = "equidistant"

        [order2]
        H.inner = ["1"]

        D1.inner_stencil = ["-1/2", "0", "1/2"]
        D1.closure_stencils = [
            ["-1", "1"],
        ]

        d1.closure = ["-3/2", "2", "-1/2"]

        [order4]
        H.closure = ["17/48", "59/48", "43/48", "49/48"]

        D2.inner_stencil = ["-1/12","4/3","-5/2","4/3","-1/12"]
        D2.closure_stencils = [
            [     "2",    "-5",      "4",       "-1",     "0",     "0"],
            [     "1",    "-2",      "1",        "0",     "0",     "0"],
            [ "-4/43", "59/43", "-110/43",   "59/43", "-4/43",     "0"],
            [ "-1/49",     "0",   "59/49", "-118/49", "64/49", "-4/49"],
        ]
    """

    parsed_toml = TOML.parse(toml_str)
    @testset "get_stencil" begin
        @test get_stencil(parsed_toml, "order2", "D1", "inner_stencil") == Stencil((-1/2, 0., 1/2), center=2)
        @test get_stencil(parsed_toml, "order2", "D1", "inner_stencil", center=1) == Stencil((-1/2, 0., 1/2); center=1)
        @test get_stencil(parsed_toml, "order2", "D1", "inner_stencil", center=3) == Stencil((-1/2, 0., 1/2); center=3)

        @test get_stencil(parsed_toml, "order2", "H", "inner") == Stencil((1.,), center=1)

        @test_throws AssertionError get_stencil(parsed_toml, "meta", "type")
        @test_throws AssertionError get_stencil(parsed_toml, "order2", "D1", "closure_stencils")
    end

    @testset "get_stencils" begin
        @test get_stencils(parsed_toml, "order2", "D1", "closure_stencils", centers=(1,)) == (Stencil((-1., 1.), center=1),)
        @test get_stencils(parsed_toml, "order2", "D1", "closure_stencils", centers=(2,)) == (Stencil((-1., 1.), center=2),)
        @test get_stencils(parsed_toml, "order2", "D1", "closure_stencils", centers=[2]) == (Stencil((-1., 1.), center=2),)

        @test get_stencils(parsed_toml, "order4", "D2", "closure_stencils",centers=[1,1,1,1]) == (
            Stencil((    2.,    -5.,      4.,     -1.,    0.,    0.), center=1),
            Stencil((    1.,    -2.,      1.,      0.,    0.,    0.), center=1),
            Stencil(( -4/43,  59/43, -110/43,   59/43, -4/43,    0.), center=1),
            Stencil(( -1/49,     0.,   59/49, -118/49, 64/49, -4/49), center=1),
        )

        @test get_stencils(parsed_toml, "order4", "D2", "closure_stencils",centers=(4,2,3,1)) == (
            Stencil((    2.,    -5.,      4.,     -1.,    0.,    0.), center=4),
            Stencil((    1.,    -2.,      1.,      0.,    0.,    0.), center=2),
            Stencil(( -4/43,  59/43, -110/43,   59/43, -4/43,    0.), center=3),
            Stencil(( -1/49,     0.,   59/49, -118/49, 64/49, -4/49), center=1),
        )

        @test get_stencils(parsed_toml, "order4", "D2", "closure_stencils",centers=1:4) == (
            Stencil((    2.,    -5.,      4.,     -1.,    0.,    0.), center=1),
            Stencil((    1.,    -2.,      1.,      0.,    0.,    0.), center=2),
            Stencil(( -4/43,  59/43, -110/43,   59/43, -4/43,    0.), center=3),
            Stencil(( -1/49,     0.,   59/49, -118/49, 64/49, -4/49), center=4),
        )

        @test_throws AssertionError get_stencils(parsed_toml, "order4", "D2", "closure_stencils",centers=(1,2,3))
        @test_throws AssertionError get_stencils(parsed_toml, "order4", "D2", "closure_stencils",centers=(1,2,3,5,4))
        @test_throws AssertionError get_stencils(parsed_toml, "order4", "D2", "inner_stencil",centers=(1,2))
    end

    @testset "get_tuple" begin
        @test get_tuple(parsed_toml, "order2", "d1", "closure") == (-3/2, 2, -1/2)

        @test_throws AssertionError get_tuple(parsed_toml, "meta", "type")
    end
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

@testset "DiagonalQuadrature" begin
    op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
    Lx = π/2.
    Ly = Float64(π)
    g_1D = EquidistantGrid(77, 0.0, Lx)
    g_2D = EquidistantGrid((77,66), (0.0, 0.0), (Lx,Ly))
    integral(H,v) = sum(H*v)
    @testset "Constructors" begin
        # 1D
        H_x = DiagonalQuadrature(spacing(g_1D)[1],op.quadratureClosure,size(g_1D));
        @test H_x == DiagonalQuadrature(g_1D,op.quadratureClosure)
        @test H_x == diagonal_quadrature(g_1D,op.quadratureClosure)
        @test H_x isa TensorMapping{T,1,1} where T
        @test H_x' isa TensorMapping{T,1,1} where T
        # 2D
        H_xy = diagonal_quadrature(g_2D,op.quadratureClosure)
        @test H_xy isa TensorMappingComposition
        @test H_xy isa TensorMapping{T,2,2} where T
        @test H_xy' isa TensorMapping{T,2,2} where T
    end

    @testset "Sizes" begin
        # 1D
        H_x = diagonal_quadrature(g_1D,op.quadratureClosure)
        @test domain_size(H_x) == size(g_1D)
        @test range_size(H_x) == size(g_1D)
        # 2D
        H_xy = diagonal_quadrature(g_2D,op.quadratureClosure)
        @test domain_size(H_xy) == size(g_2D)
        @test range_size(H_xy) == size(g_2D)
    end

    @testset "Application" begin
        # 1D
        H_x = diagonal_quadrature(g_1D,op.quadratureClosure)
        a = 3.2
        v_1D = a*ones(Float64, size(g_1D))
        u_1D = evalOn(g_1D,x->sin(x))
        @test integral(H_x,v_1D) ≈ a*Lx rtol = 1e-13
        @test integral(H_x,u_1D) ≈ 1. rtol = 1e-8
        @test H_x*v_1D == H_x'*v_1D
        # 2D
        H_xy = diagonal_quadrature(g_2D,op.quadratureClosure)
        b = 2.1
        v_2D = b*ones(Float64, size(g_2D))
        u_2D = evalOn(g_2D,(x,y)->sin(x)+cos(y))
        @test integral(H_xy,v_2D) ≈ b*Lx*Ly rtol = 1e-13
        @test integral(H_xy,u_2D) ≈ π rtol = 1e-8
        @test H_xy*v_2D ≈ H_xy'*v_2D rtol = 1e-16 #Failed for exact equality. Must differ in operation order for some reason?
    end

    @testset "Accuracy" begin
        v = ()
        for i = 0:4
            f_i(x) = 1/factorial(i)*x^i
            v = (v...,evalOn(g_1D,f_i))
        end
        # TODO: Bug in readOperator for 2nd order
        # # 2nd order
        # op2 = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=2)
        # H2 = diagonal_quadrature(g_1D,op2.quadratureClosure)
        # for i = 1:3
        #     @test integral(H2,v[i]) ≈ v[i+1] rtol = 1e-14
        # end

        # 4th order
        op4 = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        H4 = diagonal_quadrature(g_1D,op4.quadratureClosure)
        for i = 1:4
            @test integral(H4,v[i]) ≈ v[i+1][end] -  v[i+1][1] rtol = 1e-14
        end
    end

    @testset "Inferred" begin
        H_x = diagonal_quadrature(g_1D,op.quadratureClosure)
        H_xy = diagonal_quadrature(g_2D,op.quadratureClosure)
        v_1D = ones(Float64, size(g_1D))
        v_2D = ones(Float64, size(g_2D))
        @inferred H_x*v_1D
        @inferred H_x'*v_1D
        @inferred H_xy*v_2D
        @inferred H_xy'*v_2D
    end
end

@testset "InverseDiagonalQuadrature" begin
    op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
    Lx = π/2.
    Ly = Float64(π)
    g_1D = EquidistantGrid(77, 0.0, Lx)
    g_2D = EquidistantGrid((77,66), (0.0, 0.0), (Lx,Ly))
    @testset "Constructors" begin
        # 1D
        Hi_x = InverseDiagonalQuadrature(inverse_spacing(g_1D)[1], 1. ./ op.quadratureClosure, size(g_1D));
        @test Hi_x == InverseDiagonalQuadrature(g_1D,op.quadratureClosure)
        @test Hi_x == inverse_diagonal_quadrature(g_1D,op.quadratureClosure)
        @test Hi_x isa TensorMapping{T,1,1} where T
        @test Hi_x' isa TensorMapping{T,1,1} where T

        # 2D
        Hi_xy = inverse_diagonal_quadrature(g_2D,op.quadratureClosure)
        @test Hi_xy isa TensorMappingComposition
        @test Hi_xy isa TensorMapping{T,2,2} where T
        @test Hi_xy' isa TensorMapping{T,2,2} where T
    end

    @testset "Sizes" begin
        # 1D
        Hi_x = inverse_diagonal_quadrature(g_1D,op.quadratureClosure)
        @test domain_size(Hi_x) == size(g_1D)
        @test range_size(Hi_x) == size(g_1D)
        # 2D
        Hi_xy = inverse_diagonal_quadrature(g_2D,op.quadratureClosure)
        @test domain_size(Hi_xy) == size(g_2D)
        @test range_size(Hi_xy) == size(g_2D)
    end

    @testset "Application" begin
        # 1D
        H_x = diagonal_quadrature(g_1D,op.quadratureClosure)
        Hi_x = inverse_diagonal_quadrature(g_1D,op.quadratureClosure)
        v_1D = evalOn(g_1D,x->sin(x))
        u_1D = evalOn(g_1D,x->x^3-x^2+1)
        @test Hi_x*H_x*v_1D ≈ v_1D rtol = 1e-15
        @test Hi_x*H_x*u_1D ≈ u_1D rtol = 1e-15
        @test Hi_x*v_1D == Hi_x'*v_1D
        # 2D
        H_xy = diagonal_quadrature(g_2D,op.quadratureClosure)
        Hi_xy = inverse_diagonal_quadrature(g_2D,op.quadratureClosure)
        v_2D = evalOn(g_2D,(x,y)->sin(x)+cos(y))
        u_2D = evalOn(g_2D,(x,y)->x*y + x^5 - sqrt(y))
        @test Hi_xy*H_xy*v_2D ≈ v_2D rtol = 1e-15
        @test Hi_xy*H_xy*u_2D ≈ u_2D rtol = 1e-15
        @test Hi_xy*v_2D ≈ Hi_xy'*v_2D rtol = 1e-16 #Failed for exact equality. Must differ in operation order for some reason?
    end

    @testset "Inferred" begin
        Hi_x = inverse_diagonal_quadrature(g_1D,op.quadratureClosure)
        Hi_xy = inverse_diagonal_quadrature(g_2D,op.quadratureClosure)
        v_1D = ones(Float64, size(g_1D))
        v_2D = ones(Float64, size(g_2D))
        @inferred Hi_x*v_1D
        @inferred Hi_x'*v_1D
        @inferred Hi_xy*v_2D
        @inferred Hi_xy'*v_2D
    end
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
