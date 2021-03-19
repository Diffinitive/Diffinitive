using Test

using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.LazyTensors
using Sbplib.RegionIndices
using Sbplib.StaticDicts

@testset "Laplace" begin
    g_1D = EquidistantGrid(101, 0.0, 1.)
    g_3D = EquidistantGrid((51,101,52), (0.0, -1.0, 0.0), (1., 1., 1.))
    op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
    @testset "Constructors" begin

        @testset "1D" begin
            # Create all tensor mappings included in Laplace
            Δ = laplace(g_1D, op.innerStencil, op.closureStencils)
            H = inner_product(g_1D, op.quadratureClosure)
            Hi = inverse_inner_product(g_1D, op.quadratureClosure)

            (id_l, id_r) = boundary_identifiers(g_1D)

            e_l = boundary_restriction(g_1D,op.eClosure,id_l)
            e_r = boundary_restriction(g_1D,op.eClosure,id_r)
            e_dict = StaticDict(id_l => e_l, id_r => e_r)

            d_l = normal_derivative(g_1D,op.dClosure,id_l)
            d_r = normal_derivative(g_1D,op.dClosure,id_r)
            d_dict = StaticDict(id_l => d_l, id_r => d_r)

            H_l = inner_product(boundary_grid(g_1D,id_l),op.quadratureClosure)
            H_r = inner_product(boundary_grid(g_1D,id_r),op.quadratureClosure)
            Hb_dict = StaticDict(id_l => H_l, id_r => H_r)

            L = Laplace(g_1D, sbp_operators_path()*"standard_diagonal.toml"; order=4)
            @test L == Laplace(Δ,H,Hi,e_dict,d_dict,Hb_dict)
            @test L isa TensorMapping{T,1,1}  where T
            @inferred Laplace(Δ,H,Hi,e_dict,d_dict,Hb_dict)
        end
        @testset "3D" begin
            # Create all tensor mappings included in Laplace
            Δ = laplace(g_3D, op.innerStencil, op.closureStencils)
            H = inner_product(g_3D, op.quadratureClosure)
            Hi = inverse_inner_product(g_3D, op.quadratureClosure)

            (id_l, id_r, id_s, id_n, id_b, id_t) = boundary_identifiers(g_3D)
            e_l = boundary_restriction(g_3D,op.eClosure,id_l)
            e_r = boundary_restriction(g_3D,op.eClosure,id_r)
            e_s = boundary_restriction(g_3D,op.eClosure,id_s)
            e_n = boundary_restriction(g_3D,op.eClosure,id_n)
            e_b = boundary_restriction(g_3D,op.eClosure,id_b)
            e_t = boundary_restriction(g_3D,op.eClosure,id_t)
            e_dict = StaticDict(id_l => e_l, id_r => e_r,
                                id_s => e_s, id_n => e_n,
                                id_b => e_b, id_t => e_t)

            d_l = normal_derivative(g_3D,op.dClosure,id_l)
            d_r = normal_derivative(g_3D,op.dClosure,id_r)
            d_s = normal_derivative(g_3D,op.dClosure,id_s)
            d_n = normal_derivative(g_3D,op.dClosure,id_n)
            d_b = normal_derivative(g_3D,op.dClosure,id_b)
            d_t = normal_derivative(g_3D,op.dClosure,id_t)
            d_dict = StaticDict(id_l => d_l, id_r => d_r,
                                id_s => d_s, id_n => d_n,
                                id_b => d_b, id_t => d_t)

            H_l = inner_product(boundary_grid(g_3D,id_l),op.quadratureClosure)
            H_r = inner_product(boundary_grid(g_3D,id_r),op.quadratureClosure)
            H_s = inner_product(boundary_grid(g_3D,id_s),op.quadratureClosure)
            H_n = inner_product(boundary_grid(g_3D,id_n),op.quadratureClosure)
            H_b = inner_product(boundary_grid(g_3D,id_b),op.quadratureClosure)
            H_t = inner_product(boundary_grid(g_3D,id_t),op.quadratureClosure)
            Hb_dict = StaticDict(id_l => H_l, id_r => H_r,
                                 id_s => H_s, id_n => H_n,
                                 id_b => H_b, id_t => H_t)

            L = Laplace(g_3D, sbp_operators_path()*"standard_diagonal.toml"; order=4)
            @test L == Laplace(Δ,H,Hi,e_dict,d_dict,Hb_dict)
            @test L isa TensorMapping{T,3,3} where T
            @inferred Laplace(Δ,H,Hi,e_dict,d_dict,Hb_dict)
        end
    end

    @testset "laplace" begin
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
            @test L isa TensorMapping{T,3,3} where T
        end
    end

    @testset "inner_product" begin
        L = Laplace(g_3D, sbp_operators_path()*"standard_diagonal.toml"; order=4)
        @test inner_product(L) == inner_product(g_3D,op.quadratureClosure)
    end

    @testset "inverse_inner_product" begin
        L = Laplace(g_3D, sbp_operators_path()*"standard_diagonal.toml"; order=4)
        @test inverse_inner_product(L) == inverse_inner_product(g_3D,op.quadratureClosure)
    end

    @testset "boundary_restriction" begin
        L = Laplace(g_3D, sbp_operators_path()*"standard_diagonal.toml"; order=4)
        (id_l, id_r, id_s, id_n, id_b, id_t) = boundary_identifiers(g_3D)
        ids = boundary_identifiers(g_3D)
        @test boundary_restriction(L,id_l) == boundary_restriction(g_3D,op.eClosure,id_l)
        @test boundary_restriction(L,id_r) == boundary_restriction(g_3D,op.eClosure,id_r)
        @test boundary_restriction(L,id_s) == boundary_restriction(g_3D,op.eClosure,id_s)
        @test boundary_restriction(L,id_n) == boundary_restriction(g_3D,op.eClosure,id_n)
        @test boundary_restriction(L,id_b) == boundary_restriction(g_3D,op.eClosure,id_b)
        @test boundary_restriction(L,id_t) == boundary_restriction(g_3D,op.eClosure,id_t)
    end

    @testset "normal_derivative" begin
        L = Laplace(g_3D, sbp_operators_path()*"standard_diagonal.toml"; order=4)
        (id_l, id_r, id_s, id_n, id_b, id_t) = boundary_identifiers(g_3D)
        @test normal_derivative(L,id_l) == normal_derivative(g_3D,op.dClosure,id_l)
        @test normal_derivative(L,id_r) == normal_derivative(g_3D,op.dClosure,id_r)
        @test normal_derivative(L,id_s) == normal_derivative(g_3D,op.dClosure,id_s)
        @test normal_derivative(L,id_n) == normal_derivative(g_3D,op.dClosure,id_n)
        @test normal_derivative(L,id_b) == normal_derivative(g_3D,op.dClosure,id_b)
        @test normal_derivative(L,id_t) == normal_derivative(g_3D,op.dClosure,id_t)
    end

    @testset "boundary_quadrature" begin
        L = Laplace(g_3D, sbp_operators_path()*"standard_diagonal.toml"; order=4)
        (id_l, id_r, id_s, id_n, id_b, id_t) = boundary_identifiers(g_3D)
        @test boundary_quadrature(L,id_l) == inner_product(boundary_grid(g_3D,id_l),op.quadratureClosure)
        @test boundary_quadrature(L,id_r) == inner_product(boundary_grid(g_3D,id_r),op.quadratureClosure)
        @test boundary_quadrature(L,id_s) == inner_product(boundary_grid(g_3D,id_s),op.quadratureClosure)
        @test boundary_quadrature(L,id_n) == inner_product(boundary_grid(g_3D,id_n),op.quadratureClosure)
        @test boundary_quadrature(L,id_b) == inner_product(boundary_grid(g_3D,id_b),op.quadratureClosure)
        @test boundary_quadrature(L,id_t) == inner_product(boundary_grid(g_3D,id_t),op.quadratureClosure)
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
            L = Laplace(g_3D, sbp_operators_path()*"standard_diagonal.toml"; order=2)
            @test L*polynomials[1] ≈ zeros(Float64, size(g_3D)...) atol = 5e-9
            @test L*polynomials[2] ≈ zeros(Float64, size(g_3D)...) atol = 5e-9
            @test L*polynomials[3] ≈ polynomials[1] atol = 5e-9
            @test L*v ≈ Δv rtol = 5e-2 norm = l2
        end

        # 4th order interior stencil, 2nd order boundary stencil,
        # implies that L*v should be exact for binomials up to order 3.
        @testset "4th order" begin
            L = Laplace(g_3D, sbp_operators_path()*"standard_diagonal.toml"; order=4)
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
