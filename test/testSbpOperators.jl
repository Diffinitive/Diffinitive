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
import Sbplib.SbpOperators.even
import Sbplib.SbpOperators.odd


@testset "SbpOperators" begin

@testset "Stencil" begin
    s = Stencil((-2,2), (1.,2.,2.,3.,4.))
    @test s isa Stencil{Float64, 5}

    @test eltype(s) == Float64
    @test SbpOperators.scale(s, 2) == Stencil((-2,2), (2.,4.,4.,6.,8.))

    @test Stencil(1,2,3,4; center=1) == Stencil((0, 3),(1,2,3,4))
    @test Stencil(1,2,3,4; center=2) == Stencil((-1, 2),(1,2,3,4))
    @test Stencil(1,2,3,4; center=4) == Stencil((-3, 0),(1,2,3,4))

    @test CenteredStencil(1,2,3,4,5) == Stencil((-2, 2), (1,2,3,4,5))
    @test_throws ArgumentError CenteredStencil(1,2,3,4)
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
        @test get_stencil(parsed_toml, "order2", "D1", "inner_stencil") == Stencil(-1/2, 0., 1/2, center=2)
        @test get_stencil(parsed_toml, "order2", "D1", "inner_stencil", center=1) == Stencil(-1/2, 0., 1/2; center=1)
        @test get_stencil(parsed_toml, "order2", "D1", "inner_stencil", center=3) == Stencil(-1/2, 0., 1/2; center=3)

        @test get_stencil(parsed_toml, "order2", "H", "inner") == Stencil(1.; center=1)

        @test_throws AssertionError get_stencil(parsed_toml, "meta", "type")
        @test_throws AssertionError get_stencil(parsed_toml, "order2", "D1", "closure_stencils")
    end

    @testset "get_stencils" begin
        @test get_stencils(parsed_toml, "order2", "D1", "closure_stencils", centers=(1,)) == (Stencil(-1., 1., center=1),)
        @test get_stencils(parsed_toml, "order2", "D1", "closure_stencils", centers=(2,)) == (Stencil(-1., 1., center=2),)
        @test get_stencils(parsed_toml, "order2", "D1", "closure_stencils", centers=[2]) == (Stencil(-1., 1., center=2),)

        @test get_stencils(parsed_toml, "order4", "D2", "closure_stencils",centers=[1,1,1,1]) == (
            Stencil(    2.,    -5.,      4.,     -1.,    0.,    0., center=1),
            Stencil(    1.,    -2.,      1.,      0.,    0.,    0., center=1),
            Stencil( -4/43,  59/43, -110/43,   59/43, -4/43,    0., center=1),
            Stencil( -1/49,     0.,   59/49, -118/49, 64/49, -4/49, center=1),
        )

        @test get_stencils(parsed_toml, "order4", "D2", "closure_stencils",centers=(4,2,3,1)) == (
            Stencil(    2.,    -5.,      4.,     -1.,    0.,    0., center=4),
            Stencil(    1.,    -2.,      1.,      0.,    0.,    0., center=2),
            Stencil( -4/43,  59/43, -110/43,   59/43, -4/43,    0., center=3),
            Stencil( -1/49,     0.,   59/49, -118/49, 64/49, -4/49, center=1),
        )

        @test get_stencils(parsed_toml, "order4", "D2", "closure_stencils",centers=1:4) == (
            Stencil(    2.,    -5.,      4.,     -1.,    0.,    0., center=1),
            Stencil(    1.,    -2.,      1.,      0.,    0.,    0., center=2),
            Stencil( -4/43,  59/43, -110/43,   59/43, -4/43,    0., center=3),
            Stencil( -1/49,     0.,   59/49, -118/49, 64/49, -4/49, center=4),
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

@testset "VolumeOperator" begin
    inner_stencil = CenteredStencil(1/4, 2/4, 1/4)
    closure_stencils = (Stencil(1/2, 1/2; center=1), Stencil(0.,1.; center=2))
    g_1D = EquidistantGrid(11,0.,1.)
    g_2D = EquidistantGrid((11,12),(0.,0.),(1.,1.))
    g_3D = EquidistantGrid((11,12,10),(0.,0.,0.),(1.,1.,1.))
    @testset "Constructors" begin
        @testset "1D" begin
            op = VolumeOperator(inner_stencil,closure_stencils,(11,),even)
            @test op == VolumeOperator(g_1D,inner_stencil,closure_stencils,even)
            @test op == volume_operator(g_1D,inner_stencil,closure_stencils,even,1)
            @test op isa TensorMapping{T,1,1} where T
        end
        @testset "2D" begin
            op_x = volume_operator(g_2D,inner_stencil,closure_stencils,even,1)
            op_y = volume_operator(g_2D,inner_stencil,closure_stencils,even,2)
            Ix = IdentityMapping{Float64}((11,))
            Iy = IdentityMapping{Float64}((12,))
            @test op_x == VolumeOperator(inner_stencil,closure_stencils,(11,),even)⊗Iy
            @test op_y == Ix⊗VolumeOperator(inner_stencil,closure_stencils,(12,),even)
            @test op_x isa TensorMapping{T,2,2} where T
            @test op_y isa TensorMapping{T,2,2} where T
        end
        @testset "3D" begin
            op_x = volume_operator(g_3D,inner_stencil,closure_stencils,even,1)
            op_y = volume_operator(g_3D,inner_stencil,closure_stencils,even,2)
            op_z = volume_operator(g_3D,inner_stencil,closure_stencils,even,3)
            Ix = IdentityMapping{Float64}((11,))
            Iy = IdentityMapping{Float64}((12,))
            Iz = IdentityMapping{Float64}((10,))
            @test op_x == VolumeOperator(inner_stencil,closure_stencils,(11,),even)⊗Iy⊗Iz
            @test op_y == Ix⊗VolumeOperator(inner_stencil,closure_stencils,(12,),even)⊗Iz
            @test op_z == Ix⊗Iy⊗VolumeOperator(inner_stencil,closure_stencils,(10,),even)
            @test op_x isa TensorMapping{T,3,3} where T
            @test op_y isa TensorMapping{T,3,3} where T
            @test op_z isa TensorMapping{T,3,3} where T
        end
    end

    @testset "Sizes" begin
        @testset "1D" begin
            op = volume_operator(g_1D,inner_stencil,closure_stencils,even,1)
            @test range_size(op) == domain_size(op) == size(g_1D)
        end

        @testset "2D" begin
            op_x = volume_operator(g_2D,inner_stencil,closure_stencils,even,1)
            op_y = volume_operator(g_2D,inner_stencil,closure_stencils,even,2)
            @test range_size(op_y) == domain_size(op_y) ==
                  range_size(op_x) == domain_size(op_x) == size(g_2D)
        end
        @testset "3D" begin
            op_x = volume_operator(g_3D,inner_stencil,closure_stencils,even,1)
            op_y = volume_operator(g_3D,inner_stencil,closure_stencils,even,2)
            op_z = volume_operator(g_3D,inner_stencil,closure_stencils,even,3)
            @test range_size(op_z) == domain_size(op_z) ==
                  range_size(op_y) == domain_size(op_y) ==
                  range_size(op_x) == domain_size(op_x) == size(g_3D)
        end
    end

    op_x = volume_operator(g_2D,inner_stencil,closure_stencils,even,1)
    op_y = volume_operator(g_2D,inner_stencil,closure_stencils,odd,2)
    v = zeros(size(g_2D))
    Nx = size(g_2D)[1]
    Ny = size(g_2D)[2]
    for i = 1:Nx
        v[i,:] .= i
    end
    rx = copy(v)
    rx[1,:] .= 1.5
    rx[Nx,:] .= (2*Nx-1)/2
    ry = copy(v)
    ry[:,Ny-1:Ny] = -v[:,Ny-1:Ny]

    @testset "Application" begin
        @test op_x*v ≈ rx rtol = 1e-14
        @test op_y*v ≈ ry rtol = 1e-14
    end

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

    @testset "Constructors" begin
        @testset "1D" begin
            Dₓₓ = second_derivative(g_1D,op.innerStencil,op.closureStencils)
            @test Dₓₓ == second_derivative(g_1D,op.innerStencil,op.closureStencils,1)
            @test Dₓₓ isa VolumeOperator
        end
        @testset "2D" begin
            Dₓₓ = second_derivative(g_2D,op.innerStencil,op.closureStencils,1)
            D2 = second_derivative(g_1D,op.innerStencil,op.closureStencils)
            I = IdentityMapping{Float64}(size(g_2D)[2])
            @test Dₓₓ == D2⊗I
            @test Dₓₓ isa TensorMapping{T,2,2} where T
        end
    end

    # Exact differentiation is measured point-wise. In other cases
    # the error is measured in the l2-norm.
    @testset "Accuracy" begin
        @testset "1D" begin
            l2(v) = sqrt(spacing(g_1D)[1]*sum(v.^2));
            monomials = ()
            maxOrder = 4;
            for i = 0:maxOrder-1
                f_i(x) = 1/factorial(i)*x^i
                monomials = (monomials...,evalOn(g_1D,f_i))
            end
            v = evalOn(g_1D,x -> sin(x))
            vₓₓ = evalOn(g_1D,x -> -sin(x))

            # 2nd order interior stencil, 1nd order boundary stencil,
            # implies that L*v should be exact for monomials up to order 2.
            @testset "2nd order" begin
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=2)
                Dₓₓ = second_derivative(g_1D,op.innerStencil,op.closureStencils)
                @test Dₓₓ*monomials[1] ≈ zeros(Float64,size(g_1D)...) atol = 5e-10
                @test Dₓₓ*monomials[2] ≈ zeros(Float64,size(g_1D)...) atol = 5e-10
                @test Dₓₓ*monomials[3] ≈ monomials[1] atol = 5e-10
                @test Dₓₓ*v ≈ vₓₓ rtol = 5e-2 norm = l2
            end

            # 4th order interior stencil, 2nd order boundary stencil,
            # implies that L*v should be exact for monomials up to order 3.
            @testset "4th order" begin
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
                Dₓₓ = second_derivative(g_1D,op.innerStencil,op.closureStencils)
                # NOTE: high tolerances for checking the "exact" differentiation
                # due to accumulation of round-off errors/cancellation errors?
                @test Dₓₓ*monomials[1] ≈ zeros(Float64,size(g_1D)...) atol = 5e-10
                @test Dₓₓ*monomials[2] ≈ zeros(Float64,size(g_1D)...) atol = 5e-10
                @test Dₓₓ*monomials[3] ≈ monomials[1] atol = 5e-10
                @test Dₓₓ*monomials[4] ≈ monomials[2] atol = 5e-10
                @test Dₓₓ*v ≈ vₓₓ rtol = 5e-4 norm = l2
            end
        end

        @testset "2D" begin
            l2(v) = sqrt(prod(spacing(g_2D))*sum(v.^2));
            binomials = ()
            maxOrder = 4;
            for i = 0:maxOrder-1
                f_i(x,y) = 1/factorial(i)*y^i + x^i
                binomials = (binomials...,evalOn(g_2D,f_i))
            end
            v = evalOn(g_2D, (x,y) -> sin(x)+cos(y))
            v_yy = evalOn(g_2D,(x,y) -> -cos(y))

            # 2nd order interior stencil, 1st order boundary stencil,
            # implies that L*v should be exact for binomials up to order 2.
            @testset "2nd order" begin
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=2)
                Dyy = second_derivative(g_2D,op.innerStencil,op.closureStencils,2)
                @test Dyy*binomials[1] ≈ zeros(Float64,size(g_2D)...) atol = 5e-9
                @test Dyy*binomials[2] ≈ zeros(Float64,size(g_2D)...) atol = 5e-9
                @test Dyy*binomials[3] ≈ evalOn(g_2D,(x,y)->1.) atol = 5e-9
                @test Dyy*v ≈ v_yy rtol = 5e-2 norm = l2
            end

            # 4th order interior stencil, 2nd order boundary stencil,
            # implies that L*v should be exact for binomials up to order 3.
            @testset "4th order" begin
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
                Dyy = second_derivative(g_2D,op.innerStencil,op.closureStencils,2)
                # NOTE: high tolerances for checking the "exact" differentiation
                # due to accumulation of round-off errors/cancellation errors?
                @test Dyy*binomials[1] ≈ zeros(Float64,size(g_2D)...) atol = 5e-9
                @test Dyy*binomials[2] ≈ zeros(Float64,size(g_2D)...) atol = 5e-9
                @test Dyy*binomials[3] ≈ evalOn(g_2D,(x,y)->1.) atol = 5e-9
                @test Dyy*binomials[4] ≈ evalOn(g_2D,(x,y)->y) atol = 5e-9
                @test Dyy*v ≈ v_yy rtol = 5e-4 norm = l2
            end
        end
    end
end

@testset "Laplace" begin
    g_1D = EquidistantGrid(101, 0.0, 1.)
    g_3D = EquidistantGrid((51,101,52), (0.0, -1.0, 0.0), (1., 1., 1.))
    @testset "Constructors" begin
        op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        @testset "1D" begin
            L = laplace(g_1D, op.innerStencil, op.closureStencils)
            @test L == second_derivative(g_1D, op.innerStencil, op.closureStencils)
            @test L isa TensorMapping{T,1,1}  where T
        end
        @testset "3D" begin
            L = laplace(g_3D, op.innerStencil, op.closureStencils)
            @test L isa TensorMapping{T,3,3} where T
            Dxx = second_derivative(g_3D, op.innerStencil, op.closureStencils,1)
            Dyy = second_derivative(g_3D, op.innerStencil, op.closureStencils,2)
            Dzz = second_derivative(g_3D, op.innerStencil, op.closureStencils,3)
            @test L == Dxx + Dyy + Dzz
        end
    end

    # Exact differentiation is measured point-wise. In other cases
    # the error is measured in the l2-norm.
    @testset "Accuracy" begin
        l2(v) = sqrt(prod(spacing(g_3D))*sum(v.^2));
        polynomials = ()
        maxOrder = 4;
        for i = 0:maxOrder-1
            f_i(x,y,z) = 1/factorial(i)*(y^i + x^i + z^i)
            polynomials = (polynomials...,evalOn(g_3D,f_i))
        end
        v = evalOn(g_3D, (x,y,z) -> sin(x) + cos(y) + exp(z))
        Δv = evalOn(g_3D,(x,y,z) -> -sin(x) - cos(y) + exp(z))

        # 2nd order interior stencil, 1st order boundary stencil,
        # implies that L*v should be exact for binomials up to order 2.
        @testset "2nd order" begin
            op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=2)
            L = laplace(g_3D,op.innerStencil,op.closureStencils)
            @test L*polynomials[1] ≈ zeros(Float64, size(g_3D)...) atol = 5e-9
            @test L*polynomials[2] ≈ zeros(Float64, size(g_3D)...) atol = 5e-9
            @test L*polynomials[3] ≈ polynomials[1] atol = 5e-9
            @test L*v ≈ Δv rtol = 5e-2 norm = l2
        end

        # 4th order interior stencil, 2nd order boundary stencil,
        # implies that L*v should be exact for binomials up to order 3.
        @testset "4th order" begin
            op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
            L = laplace(g_3D,op.innerStencil,op.closureStencils)
            # NOTE: high tolerances for checking the "exact" differentiation
            # due to accumulation of round-off errors/cancellation errors?
            @test L*polynomials[1] ≈ zeros(Float64, size(g_3D)...) atol = 5e-9
            @test L*polynomials[2] ≈ zeros(Float64, size(g_3D)...) atol = 5e-9
            @test L*polynomials[3] ≈ polynomials[1] atol = 5e-9
            @test L*polynomials[4] ≈ polynomials[2] atol = 5e-9
            @test L*v ≈ Δv rtol = 5e-4 norm = l2
        end
    end
end

@testset "Quadrature diagonal" begin
    Lx = π/2.
    Ly = Float64(π)
    Lz = 1.
    g_1D = EquidistantGrid(77, 0.0, Lx)
    g_2D = EquidistantGrid((77,66), (0.0, 0.0), (Lx,Ly))
    g_3D = EquidistantGrid((10,10, 10), (0.0, 0.0, 0.0), (Lx,Ly,Lz))
    integral(H,v) = sum(H*v)
    @testset "quadrature" begin
        op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        @testset "0D" begin
            H = quadrature(EquidistantGrid{Float64}(),op.quadratureClosure)
            @test H == IdentityMapping{Float64}()
            @test H isa TensorMapping{T,0,0} where T
        end
        @testset "1D" begin
            H = quadrature(g_1D,op.quadratureClosure)
            inner_stencil = CenteredStencil(1.)
            @test H == quadrature(g_1D,op.quadratureClosure,inner_stencil)
            @test H isa TensorMapping{T,1,1} where T
        end
        @testset "2D" begin
            H = quadrature(g_2D,op.quadratureClosure)
            H_x = quadrature(restrict(g_2D,1),op.quadratureClosure)
            H_y = quadrature(restrict(g_2D,2),op.quadratureClosure)
            @test H == H_x⊗H_y
            @test H isa TensorMapping{T,2,2} where T
        end
    end

    @testset "Sizes" begin
        op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        @testset "1D" begin
            H = quadrature(g_1D,op.quadratureClosure)
            @test domain_size(H) == size(g_1D)
            @test range_size(H) == size(g_1D)
        end
        @testset "2D" begin
            H = quadrature(g_2D,op.quadratureClosure)
            @test domain_size(H) == size(g_2D)
            @test range_size(H) == size(g_2D)
        end
    end

    @testset "Accuracy" begin
        @testset "1D" begin
            v = ()
            for i = 0:4
                f_i(x) = 1/factorial(i)*x^i
                v = (v...,evalOn(g_1D,f_i))
            end
            u = evalOn(g_1D,x->sin(x))

            @testset "2nd order" begin
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=2)
                H = quadrature(g_1D,op.quadratureClosure)
                for i = 1:2
                    @test integral(H,v[i]) ≈ v[i+1][end] - v[i+1][1] rtol = 1e-14
                end
                @test integral(H,u) ≈ 1. rtol = 1e-4
            end

            @testset "4th order" begin
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
                H = quadrature(g_1D,op.quadratureClosure)
                for i = 1:4
                    @test integral(H,v[i]) ≈ v[i+1][end] -  v[i+1][1] rtol = 1e-14
                end
                @test integral(H,u) ≈ 1. rtol = 1e-8
            end
        end

        @testset "2D" begin
            b = 2.1
            v = b*ones(Float64, size(g_2D))
            u = evalOn(g_2D,(x,y)->sin(x)+cos(y))
            @testset "2nd order" begin
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=2)
                H = quadrature(g_2D,op.quadratureClosure)
                @test integral(H,v) ≈ b*Lx*Ly rtol = 1e-13
                @test integral(H,u) ≈ π rtol = 1e-4
            end
            @testset "4th order" begin
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
                H = quadrature(g_2D,op.quadratureClosure)
                @test integral(H,v) ≈ b*Lx*Ly rtol = 1e-13
                @test integral(H,u) ≈ π rtol = 1e-8
            end
        end
    end
end

@testset "InverseDiagonalQuadrature" begin
    Lx = π/2.
    Ly = Float64(π)
    g_1D = EquidistantGrid(77, 0.0, Lx)
    g_2D = EquidistantGrid((77,66), (0.0, 0.0), (Lx,Ly))
    @testset "Constructors" begin
        op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        @testset "1D" begin
            Hi = InverseDiagonalQuadrature(g_1D, op.quadratureClosure);
            inner_stencil = CenteredStencil(1.)
            closures = ()
            for i = 1:length(op.quadratureClosure)
                closures = (closures...,Stencil(op.quadratureClosure[i].range,1.0./op.quadratureClosure[i].weights))
            end
            @test Hi == InverseQuadrature(g_1D,inner_stencil,closures)
            @test Hi isa TensorMapping{T,1,1} where T
        end
        @testset "2D" begin
            Hi = InverseDiagonalQuadrature(g_2D,op.quadratureClosure)
            Hi_x = InverseDiagonalQuadrature(restrict(g_2D,1),op.quadratureClosure)
            Hi_y = InverseDiagonalQuadrature(restrict(g_2D,2),op.quadratureClosure)
            @test Hi == Hi_x⊗Hi_y
            @test Hi isa TensorMapping{T,2,2} where T
        end
    end

    @testset "Sizes" begin
        op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        @testset "1D" begin
            Hi = InverseDiagonalQuadrature(g_1D,op.quadratureClosure)
            @test domain_size(Hi) == size(g_1D)
            @test range_size(Hi) == size(g_1D)
        end
        @testset "2D" begin
            Hi = InverseDiagonalQuadrature(g_2D,op.quadratureClosure)
            @test domain_size(Hi) == size(g_2D)
            @test range_size(Hi) == size(g_2D)
        end
    end

    @testset "Accuracy" begin
        @testset "1D" begin
            v = evalOn(g_1D,x->sin(x))
            u = evalOn(g_1D,x->x^3-x^2+1)
            @testset "2nd order" begin
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=2)
                H = quadrature(g_1D,op.quadratureClosure)
                Hi = InverseDiagonalQuadrature(g_1D,op.quadratureClosure)
                @test Hi*H*v ≈ v rtol = 1e-15
                @test Hi*H*u ≈ u rtol = 1e-15
            end
            @testset "4th order" begin
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
                H = quadrature(g_1D,op.quadratureClosure)
                Hi = InverseDiagonalQuadrature(g_1D,op.quadratureClosure)
                @test Hi*H*v ≈ v rtol = 1e-15
                @test Hi*H*u ≈ u rtol = 1e-15
            end
        end
        @testset "2D" begin
            v = evalOn(g_2D,(x,y)->sin(x)+cos(y))
            u = evalOn(g_2D,(x,y)->x*y + x^5 - sqrt(y))
            @testset "2nd order" begin
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=2)
                H = quadrature(g_2D,op.quadratureClosure)
                Hi = InverseDiagonalQuadrature(g_2D,op.quadratureClosure)
                @test Hi*H*v ≈ v rtol = 1e-15
                @test Hi*H*u ≈ u rtol = 1e-15
            end
            @testset "4th order" begin
                op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
                H = quadrature(g_2D,op.quadratureClosure)
                Hi = InverseDiagonalQuadrature(g_2D,op.quadratureClosure)
                @test Hi*H*v ≈ v rtol = 1e-15
                @test Hi*H*u ≈ u rtol = 1e-15
            end
        end
    end
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
            @test op_r == BoundaryOperator(g_1D,closure_stencil,Upper())
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

@testset "boundary_restriction" begin
    op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
    g_1D = EquidistantGrid(11, 0.0, 1.0)
    g_2D = EquidistantGrid((11,15), (0.0, 0.0), (1.0,1.0))

    @testset "boundary_restriction" begin
        @testset "1D" begin
            e_l = boundary_restriction(g_1D,op.eClosure,Lower())
            @test e_l == boundary_restriction(g_1D,op.eClosure,CartesianBoundary{1,Lower}())
            @test e_l == BoundaryOperator(g_1D,op.eClosure,Lower())
            @test e_l isa BoundaryOperator{T,Lower} where T
            @test e_l isa TensorMapping{T,0,1} where T

            e_r = boundary_restriction(g_1D,op.eClosure,Upper())
            @test e_r == boundary_restriction(g_1D,op.eClosure,CartesianBoundary{1,Upper}())
            @test e_r == BoundaryOperator(g_1D,op.eClosure,Upper())
            @test e_r isa BoundaryOperator{T,Upper} where T
            @test e_r isa TensorMapping{T,0,1} where T
        end

        @testset "2D" begin
            e_w = boundary_restriction(g_2D,op.eClosure,CartesianBoundary{1,Upper}())
            @test e_w isa InflatedTensorMapping
            @test e_w isa TensorMapping{T,1,2} where T
        end
    end

    @testset "Application" begin
        @testset "1D" begin
            e_l = boundary_restriction(g_1D, op.eClosure, CartesianBoundary{1,Lower}())
            e_r = boundary_restriction(g_1D, op.eClosure, CartesianBoundary{1,Upper}())

            v = evalOn(g_1D,x->1+x^2)
            u = fill(3.124)

            @test (e_l*v)[] == v[1]
            @test (e_r*v)[] == v[end]
            @test (e_r*v)[1] == v[end]
        end

        @testset "2D" begin
            e_w = boundary_restriction(g_2D, op.eClosure, CartesianBoundary{1,Lower}())
            e_e = boundary_restriction(g_2D, op.eClosure, CartesianBoundary{1,Upper}())
            e_s = boundary_restriction(g_2D, op.eClosure, CartesianBoundary{2,Lower}())
            e_n = boundary_restriction(g_2D, op.eClosure, CartesianBoundary{2,Upper}())

            v = rand(11, 15)
            u = fill(3.124)

            @test e_w*v == v[1,:]
            @test e_e*v == v[end,:]
            @test e_s*v == v[:,1]
            @test e_n*v == v[:,end]
       end
    end
end

@testset "normal_derivative" begin
    g_1D = EquidistantGrid(11, 0.0, 1.0)
    g_2D = EquidistantGrid((11,12), (0.0, 0.0), (1.0,1.0))
    @testset "normal_derivative" begin
        op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
        @testset "1D" begin
            d_l = normal_derivative(g_1D, op.dClosure, Lower())
            @test d_l == normal_derivative(g_1D, op.dClosure, CartesianBoundary{1,Lower}())
            @test d_l isa BoundaryOperator{T,Lower} where T
            @test d_l isa TensorMapping{T,0,1} where T
        end
        @testset "2D" begin
            op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
            d_w = normal_derivative(g_2D, op.dClosure, CartesianBoundary{1,Lower}())
            d_n = normal_derivative(g_2D, op.dClosure, CartesianBoundary{2,Upper}())
            Ix = IdentityMapping{Float64}((size(g_2D)[1],))
            Iy = IdentityMapping{Float64}((size(g_2D)[2],))
            d_l = normal_derivative(restrict(g_2D,1),op.dClosure,Lower())
            d_r = normal_derivative(restrict(g_2D,2),op.dClosure,Upper())
            @test d_w ==  d_l⊗Iy
            @test d_n ==  Ix⊗d_r
            @test d_w isa TensorMapping{T,1,2} where T
            @test d_n isa TensorMapping{T,1,2} where T
        end
    end
    @testset "Accuracy" begin
        v = evalOn(g_2D, (x,y)-> x^2 + (y-1)^2 + x*y)
        v∂x = evalOn(g_2D, (x,y)-> 2*x + y)
        v∂y = evalOn(g_2D, (x,y)-> 2*(y-1) + x)
        # TODO: Test for higher order polynomials?
        @testset "2nd order" begin
            op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=2)
            d_w = normal_derivative(g_2D, op.dClosure, CartesianBoundary{1,Lower}())
            d_e = normal_derivative(g_2D, op.dClosure, CartesianBoundary{1,Upper}())
            d_s = normal_derivative(g_2D, op.dClosure, CartesianBoundary{2,Lower}())
            d_n = normal_derivative(g_2D, op.dClosure, CartesianBoundary{2,Upper}())

            @test d_w*v ≈ v∂x[1,:] atol = 1e-13
            @test d_e*v ≈ -v∂x[end,:] atol = 1e-13
            @test d_s*v ≈ v∂y[:,1] atol = 1e-13
            @test d_n*v ≈ -v∂y[:,end] atol = 1e-13
        end

        @testset "4th order" begin
            op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
            d_w = normal_derivative(g_2D, op.dClosure, CartesianBoundary{1,Lower}())
            d_e = normal_derivative(g_2D, op.dClosure, CartesianBoundary{1,Upper}())
            d_s = normal_derivative(g_2D, op.dClosure, CartesianBoundary{2,Lower}())
            d_n = normal_derivative(g_2D, op.dClosure, CartesianBoundary{2,Upper}())

            @test d_w*v ≈ v∂x[1,:] atol = 1e-13
            @test d_e*v ≈ -v∂x[end,:] atol = 1e-13
            @test d_s*v ≈ v∂y[:,1] atol = 1e-13
            @test d_n*v ≈ -v∂y[:,end] atol = 1e-13
        end
    end
end

end
