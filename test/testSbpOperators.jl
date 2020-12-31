using Test
using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.RegionIndices
using Sbplib.LazyTensors
using LinearAlgebra
using TOML

import Sbplib.SbpOperators.Stencil
import Sbplib.SbpOperators.VolumeOperator
import Sbplib.SbpOperators.volume_operator
import Sbplib.SbpOperators.BoundaryOperator
import Sbplib.SbpOperators.boundary_operator
import Sbplib.SbpOperators.Parity


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

@testset "VolumeOperator" begin
    inner_stencil = Stencil(1/4 .* (1.,2.,1.),center=2)
    closure_stencils = (Stencil(1/2 .* (1.,1.),center=1),Stencil((0.,1.),center=2))
    g_1D = EquidistantGrid(11,0.,1.)
    g_2D = EquidistantGrid((11,12),(0.,0.),(1.,1.))
    g_3D = EquidistantGrid((11,12,10),(0.,0.,0.),(1.,1.,1.))
    @testset "Constructors" begin
        #TODO: How are even and odd in SbpOperators.Parity exposed? Currently constructing even as Parity(1)
        @testset "1D" begin
            op = VolumeOperator(inner_stencil,closure_stencils,(11,),Parity(1))
            @test op == VolumeOperator(g_1D,inner_stencil,closure_stencils,Parity(1))
            @test op == volume_operator(g_1D,inner_stencil,closure_stencils,Parity(1),1)
            @test op isa TensorMapping{T,1,1} where T
        end
        @testset "2D" begin
            op_x = volume_operator(g_2D,inner_stencil,closure_stencils,Parity(1),1)
            op_y = volume_operator(g_2D,inner_stencil,closure_stencils,Parity(1),2)
            Ix = IdentityMapping{Float64}((11,))
            Iy = IdentityMapping{Float64}((12,))
            @test op_x == VolumeOperator(inner_stencil,closure_stencils,(11,),Parity(1))⊗Iy
            @test op_y == Ix⊗VolumeOperator(inner_stencil,closure_stencils,(12,),Parity(1))
            @test op_x isa TensorMapping{T,2,2} where T
            @test op_y isa TensorMapping{T,2,2} where T
        end
        @testset "3D" begin
            op_x = volume_operator(g_3D,inner_stencil,closure_stencils,Parity(1),1)
            op_y = volume_operator(g_3D,inner_stencil,closure_stencils,Parity(1),2)
            op_z = volume_operator(g_3D,inner_stencil,closure_stencils,Parity(1),3)
            Ix = IdentityMapping{Float64}((11,))
            Iy = IdentityMapping{Float64}((12,))
            Iz = IdentityMapping{Float64}((10,))
            @test op_x == VolumeOperator(inner_stencil,closure_stencils,(11,),Parity(1))⊗Iy⊗Iz
            @test op_y == Ix⊗VolumeOperator(inner_stencil,closure_stencils,(12,),Parity(1))⊗Iz
            @test op_z == Ix⊗Iy⊗VolumeOperator(inner_stencil,closure_stencils,(10,),Parity(1))
            @test op_x isa TensorMapping{T,3,3} where T
            @test op_y isa TensorMapping{T,3,3} where T
            @test op_z isa TensorMapping{T,3,3} where T
        end
    end

    @testset "Sizes" begin
        @testset "1D" begin
            op = volume_operator(g_1D,inner_stencil,closure_stencils,Parity(1),1)
            @test range_size(op) == domain_size(op) == size(g_1D)
        end

        @testset "2D" begin
            op_x = volume_operator(g_2D,inner_stencil,closure_stencils,Parity(1),1)
            op_y = volume_operator(g_2D,inner_stencil,closure_stencils,Parity(1),2)
            @test range_size(op_y) == domain_size(op_y) ==
                  range_size(op_x) == domain_size(op_x) == size(g_2D)
        end
        @testset "3D" begin
            op_x = volume_operator(g_3D,inner_stencil,closure_stencils,Parity(1),1)
            op_y = volume_operator(g_3D,inner_stencil,closure_stencils,Parity(1),2)
            op_z = volume_operator(g_3D,inner_stencil,closure_stencils,Parity(1),3)
            @test range_size(op_z) == domain_size(op_z) ==
                  range_size(op_y) == domain_size(op_y) ==
                  range_size(op_x) == domain_size(op_x) == size(g_3D)
        end
    end

    # TODO: Test for other dimensions?
    op_x = volume_operator(g_2D,inner_stencil,closure_stencils,Parity(1),1)
    op_y = volume_operator(g_2D,inner_stencil,closure_stencils,Parity(1),2)
    v = zeros(size(g_2D))
    Nx = size(g_2D)[1]
    for i = 1:Nx
        v[i,:] .= i
    end
    rx = copy(v)
    rx[1,:] .= 1.5
    rx[end,:] .= (2*Nx-1)/2
    ry = copy(v)

    @testset "Application" begin
        @test op_x*v ≈ rx rtol = 1e-14
        @test op_y*v ≈ ry rtol = 1e-14
    end

    # TODO: Test for other dimensions?
    @testset "Regions" begin
        @test (op_x*v)[Index(1,Lower),Index(3,Interior)] ≈ rx[1,3] rtol = 1e-14
        @test (op_x*v)[Index(2,Lower),Index(3,Interior)] ≈ rx[2,3] rtol = 1e-14
        @test (op_x*v)[Index(6,Interior),Index(3,Interior)] ≈ rx[6,3] rtol = 1e-14
        @test (op_x*v)[Index(10,Upper),Index(3,Interior)] ≈ rx[10,3] rtol = 1e-14
        @test (op_x*v)[Index(11,Upper),Index(3,Interior)] ≈ rx[11,3] rtol = 1e-14

        @test_throws BoundsError (op_x*v)[Index(3,Lower),Index(3,Interior)]
        @test_throws BoundsError (op_x*v)[Index(9,Upper),Index(3,Interior)]

        @test (op_y*v)[Index(3,Interior),Index(1,Lower)] ≈ ry[3,1] rtol = 1e-14
        @test (op_y*v)[Index(3,Interior),Index(2,Lower)] ≈ ry[3,2] rtol = 1e-14
        @test (op_y*v)[Index(3,Interior),Index(6,Interior)] ≈ ry[3,6] rtol = 1e-14
        @test (op_y*v)[Index(3,Interior),Index(11,Upper)] ≈ ry[3,11] rtol = 1e-14
        @test (op_y*v)[Index(3,Interior),Index(12,Upper)] ≈ ry[3,12] rtol = 1e-14

        @test_throws BoundsError (op_y*v)[Index(3,Interior),Index(10,Upper)]
        @test_throws BoundsError (op_y*v)[Index(3,Interior),Index(3,Lower)]
    end

    # TODO: Test for other dimensions?
    @testset "Inferred" begin
        @inferred apply(op_x, v,1,1)
        @inferred apply(op_x, v, Index(1,Lower),Index(1,Lower))
        @inferred apply(op_x, v, Index(6,Interior),Index(1,Lower))
        @inferred apply(op_x, v, Index(11,Upper),Index(1,Lower))

        @inferred apply(op_y, v,1,1)
        @inferred apply(op_y, v, Index(1,Lower),Index(1,Lower))
        @inferred apply(op_y, v, Index(1,Lower),Index(6,Interior))
        @inferred apply(op_y, v, Index(1,Lower),Index(11,Upper))
    end

end

@testset "SecondDerivative" begin
    op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
    Lx = 3.5
    Ly = 3.
    g_1D = EquidistantGrid(121, 0.0, Lx)
    g_2D = EquidistantGrid((121,123), (0.0, 0.0), (Lx, Ly))

    # TODO: These areant really constructors. Better name?
    @testset "Constructors" begin
        @testset "1D" begin
            Dₓₓ = SecondDerivative(g_1D,op.innerStencil,op.closureStencils)
            @test Dₓₓ == SecondDerivative(g_1D,op.innerStencil,op.closureStencils,1)
            @test Dₓₓ isa VolumeOperator
        end
        @testset "2D" begin
            Dₓₓ = SecondDerivative(g_2D,op.innerStencil,op.closureStencils,1)
            D2 = SecondDerivative(g_1D,op.innerStencil,op.closureStencils)
            I = IdentityMapping{Float64}(size(g_2D)[2])
            @test Dₓₓ == D2⊗I
            @test Dₓₓ isa TensorMapping{T,2,2} where T
        end
    end

    @testset "Accuracy" begin
        @testset "1D" begin
            monomials = ()
            maxOrder = 4;
            for i = 0:maxOrder-1
                f_i(x) = 1/factorial(i)*x^i
                monomials = (monomials...,evalOn(g_1D,f_i))
            end
            l2(v) = sqrt(spacing(g_1D)[1]*sum(v.^2));

            #TODO: Error when reading second order stencil!
            # @testset "2nd order" begin
            #     op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=2)
            #     Dₓₓ = SecondDerivative(g_1D,op.innerStencil,op.closureStencils)
            #     @test Dₓₓ*monomials[1] ≈ zeros(Float64,size(g_1D)...) atol = 5e-13
            #     @test Dₓₓ*monomials[2] ≈ zeros(Float64,size(g_1D)...) atol = 5e-13
            # end

            # 4th order interior stencil, 2nd order boundary stencil,
            # implies that L*v should be exact for monomials up to order 3.
            # Exact differentiation is measured point-wise. For other grid functions
            # the error is measured in the l2-norm.
            @testset "4th order" begin
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
                Dₓₓ = SecondDerivative(g_1D,op.innerStencil,op.closureStencils)
                # TODO: high tolerances for checking the "exact" differentiation
                # due to accumulation of round-off errors/cancellation errors?
                @test Dₓₓ*monomials[1] ≈ zeros(Float64,size(g_1D)...) atol = 5e-10
                @test Dₓₓ*monomials[2] ≈ zeros(Float64,size(g_1D)...) atol = 5e-10
                @test Dₓₓ*monomials[3] ≈ monomials[1] atol = 5e-10
                @test Dₓₓ*monomials[4] ≈ monomials[2] atol = 5e-10
                @test Dₓₓ*evalOn(g_1D,x -> sin(x)) ≈ evalOn(g_1D,x -> -sin(x)) rtol = 5e-4 norm = l2
            end
        end

        @testset "2D" begin
            binomials = ()
            maxOrder = 4;
            for i = 0:maxOrder-1
                f_i(x,y) = 1/factorial(i)*y^i + x^i
                binomials = (binomials...,evalOn(g_2D,f_i))
            end
            l2(v) = sqrt(prod(spacing(g_2D))*sum(v.^2));

            #TODO: Error when reading second order stencil!
            # @testset "2nd order" begin
            #     op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=2)
            #     Dyy = SecondDerivative(g_2D,op.innerStencil,op.closureStencils,2)
            #     @test Dyy*binomials[1] ≈ evalOn(g_2D,(x,y)->0.) atol = 5e-12
            #     @test Dyy*binomials[2] ≈ evalOn(g_2D,(x,y)->0.) atol = 5e-12
            # end

            # 4th order interior stencil, 2nd order boundary stencil,
            # implies that L*v should be exact for binomials up to order 3.
            # Exact differentiation is measured point-wise. For other grid functions
            # the error is measured in the l2-norm.
            @testset "4th order" begin
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
                Dyy = SecondDerivative(g_2D,op.innerStencil,op.closureStencils,2)
                # TODO: high tolerances for checking the "exact" differentiation
                # due to accumulation of round-off errors/cancellation errors?
                @test Dyy*binomials[1] ≈ zeros(Float64,size(g_2D)...) atol = 5e-9
                @test Dyy*binomials[2] ≈ zeros(Float64,size(g_2D)...) atol = 5e-9
                @test Dyy*binomials[3] ≈ evalOn(g_2D,(x,y)->1.) atol = 5e-9
                @test Dyy*binomials[4] ≈ evalOn(g_2D,(x,y)->y) atol = 5e-9
                @test Dyy*evalOn(g_2D, (x,y) -> sin(x)+cos(y)) ≈ evalOn(g_2D,(x,y) -> -cos(y)) rtol = 5e-4 norm = l2
            end
        end
    end
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
    @test norm(L*v0) ≈ 0 atol=1e-9
    @test norm(L*v1) ≈ 0 atol=1e-9
    @test L*v2 ≈ v0 # Seems to be more accurate
    @test L*v3 ≈ v1 atol=1e-9

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

@testset "BoundaryOperator" begin
    closure_stencil = Stencil((0,2), (2.,1.,3.))
    g_1D = EquidistantGrid(11, 0.0, 1.0)
    g_2D = EquidistantGrid((11,15), (0.0, 0.0), (1.0,1.0))

    @testset "Constructors" begin
        @testset "1D" begin
            op_l = BoundaryOperator{Lower}(closure_stencil,size(g_1D)[1])
            @test op_l == BoundaryOperator(g_1D,closure_stencil,Lower())
            @test op_l == boundary_operator(g_1D,closure_stencil,CartesianBoundary{1,Lower}())
            @test op_l isa TensorMapping{T,0,1} where T

            op_r = BoundaryOperator{Upper}(closure_stencil,size(g_1D)[1])
            @test op_r == BoundaryRestriction(g_1D,closure_stencil,Upper())
            @test op_r == boundary_operator(g_1D,closure_stencil,CartesianBoundary{1,Upper}())
            @test op_r isa TensorMapping{T,0,1} where T
        end

        @testset "2D" begin
            e_w = boundary_operator(g_2D,closure_stencil,CartesianBoundary{1,Upper}())
            @test e_w isa InflatedTensorMapping
            @test e_w isa TensorMapping{T,1,2} where T
        end
    end

    op_l = boundary_operator(g_1D, closure_stencil, CartesianBoundary{1,Lower}())
    op_r = boundary_operator(g_1D, closure_stencil, CartesianBoundary{1,Upper}())

    op_w = boundary_operator(g_2D, closure_stencil, CartesianBoundary{1,Lower}())
    op_e = boundary_operator(g_2D, closure_stencil, CartesianBoundary{1,Upper}())
    op_s = boundary_operator(g_2D, closure_stencil, CartesianBoundary{2,Lower}())
    op_n = boundary_operator(g_2D, closure_stencil, CartesianBoundary{2,Upper}())

    @testset "Sizes" begin
        @testset "1D" begin
            @test domain_size(op_l) == (11,)
            @test domain_size(op_r) == (11,)

            @test range_size(op_l) == ()
            @test range_size(op_r) == ()
        end

        @testset "2D" begin
            @test domain_size(op_w) == (11,15)
            @test domain_size(op_e) == (11,15)
            @test domain_size(op_s) == (11,15)
            @test domain_size(op_n) == (11,15)

            @test range_size(op_w) == (15,)
            @test range_size(op_e) == (15,)
            @test range_size(op_s) == (11,)
            @test range_size(op_n) == (11,)
        end
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
        end

        @testset "2D" begin
            v = rand(size(g_2D)...)
            u = fill(3.124)
            @test op_w*v ≈ 2*v[1,:] + v[2,:] + 3*v[3,:] rtol = 1e-14
            @test op_e*v ≈ 2*v[end,:] + v[end-1,:] + 3*v[end-2,:] rtol = 1e-14
            @test op_s*v ≈ 2*v[:,1] + v[:,2] + 3*v[:,3] rtol = 1e-14
            @test op_n*v ≈ 2*v[:,end] + v[:,end-1] + 3*v[:,end-2] rtol = 1e-14


           g_x = rand(size(g_2D)[1])
           g_y = rand(size(g_2D)[2])

           G_w = zeros(Float64, size(g_2D)...)
           G_w[1,:] = 2*g_y
           G_w[2,:] = g_y
           G_w[3,:] = 3*g_y

           G_e = zeros(Float64, size(g_2D)...)
           G_e[end,:] = 2*g_y
           G_e[end-1,:] = g_y
           G_e[end-2,:] = 3*g_y

           G_s = zeros(Float64, size(g_2D)...)
           G_s[:,1] = 2*g_x
           G_s[:,2] = g_x
           G_s[:,3] = 3*g_x

           G_n = zeros(Float64, size(g_2D)...)
           G_n[:,end] = 2*g_x
           G_n[:,end-1] = g_x
           G_n[:,end-2] = 3*g_x

           @test op_w'*g_y == G_w
           @test op_e'*g_y == G_e
           @test op_s'*g_x == G_s
           @test op_n'*g_x == G_n
       end

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

@testset "BoundaryRestriction" begin
    op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
    g_1D = EquidistantGrid(11, 0.0, 1.0)
    g_2D = EquidistantGrid((11,15), (0.0, 0.0), (1.0,1.0))

    @testset "Constructors" begin
        @testset "1D" begin
            e_l = BoundaryRestriction(g_1D,op.eClosure,Lower())
            @test e_l == BoundaryRestriction(g_1D,op.eClosure,CartesianBoundary{1,Lower}())
            @test e_l == BoundaryOperator(g_1D,op.eClosure,Lower())
            @test e_l isa BoundaryOperator{T,Lower} where T
            @test e_l isa TensorMapping{T,0,1} where T

            e_r = BoundaryRestriction(g_1D,op.eClosure,Upper())
            @test e_r == BoundaryRestriction(g_1D,op.eClosure,CartesianBoundary{1,Upper}())
            @test e_r == BoundaryOperator(g_1D,op.eClosure,Upper())
            @test e_r isa BoundaryOperator{T,Upper} where T
            @test e_r isa TensorMapping{T,0,1} where T
        end

        @testset "2D" begin
            e_w = BoundaryRestriction(g_2D,op.eClosure,CartesianBoundary{1,Upper}())
            @test e_w isa InflatedTensorMapping
            @test e_w isa TensorMapping{T,1,2} where T
        end
    end

    @testset "Application" begin
        @testset "1D" begin
            e_l = BoundaryRestriction(g_1D, op.eClosure, CartesianBoundary{1,Lower}())
            e_r = BoundaryRestriction(g_1D, op.eClosure, CartesianBoundary{1,Upper}())

            v = evalOn(g_1D,x->1+x^2)
            u = fill(3.124)

            @test (e_l*v)[] == v[1]
            @test (e_r*v)[] == v[end]
            @test (e_r*v)[1] == v[end]
            @test e_l'*u == [u[]; zeros(10)]
            @test e_r'*u == [zeros(10); u[]]
        end

        @testset "2D" begin
            e_w = BoundaryRestriction(g_2D, op.eClosure, CartesianBoundary{1,Lower}())
            e_e = BoundaryRestriction(g_2D, op.eClosure, CartesianBoundary{1,Upper}())
            e_s = BoundaryRestriction(g_2D, op.eClosure, CartesianBoundary{2,Lower}())
            e_n = BoundaryRestriction(g_2D, op.eClosure, CartesianBoundary{2,Upper}())

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
    end
end

@testset "NormalDerivative" begin
    op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
    g = EquidistantGrid((5,6), (0.0, 0.0), (4.0,5.0))

    d_w = NormalDerivative(g, op.dClosure, CartesianBoundary{1,Lower}())
    d_e = NormalDerivative(g, op.dClosure, CartesianBoundary{1,Upper}())
    d_s = NormalDerivative(g, op.dClosure, CartesianBoundary{2,Lower}())
    d_n = NormalDerivative(g, op.dClosure, CartesianBoundary{2,Upper}())


    v = evalOn(g, (x,y)-> x^2 + (y-1)^2 + x*y)
    v∂x = evalOn(g, (x,y)-> 2*x + y)
    v∂y = evalOn(g, (x,y)-> 2*(y-1) + x)

    @test d_w isa TensorMapping{T,1,2} where T

    @test d_w*v ≈ v∂x[1,:]
    @test d_e*v ≈ -v∂x[end,:]
    @test d_s*v ≈ v∂y[:,1]
    @test d_n*v ≈ -v∂y[:,end]


    d_x_l = zeros(Float64, size(g)[1])
    d_x_u = zeros(Float64, size(g)[1])
    for i ∈ eachindex(d_x_l)
        d_x_l[i] = op.dClosure[i-1]
        d_x_u[i] = op.dClosure[length(d_x_u)-i]
    end

    d_y_l = zeros(Float64, size(g)[2])
    d_y_u = zeros(Float64, size(g)[2])
    for i ∈ eachindex(d_y_l)
        d_y_l[i] = op.dClosure[i-1]
        d_y_u[i] = op.dClosure[length(d_y_u)-i]
    end

    function prod_matrix(x,y)
        G = zeros(Float64, length(x), length(y))
        for I ∈ CartesianIndices(G)
            G[I] = x[I[1]]*y[I[2]]
        end

        return G
    end

    g_x = [1,2,3,4.0,5]
    g_y = [5,4,3,2,1.0,11]

    G_w = prod_matrix(d_x_l, g_y)
    G_e = prod_matrix(d_x_u, g_y)
    G_s = prod_matrix(g_x, d_y_l)
    G_n = prod_matrix(g_x, d_y_u)

    @test d_w'*g_y ≈ G_w
    @test_broken d_e'*g_y ≈ G_e
    @test d_s'*g_x ≈ G_s
    @test_broken d_n'*g_x ≈ G_n
end
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
