using Test

using Sbplib.SbpOperators
using Sbplib.Grids
using Sbplib.LazyTensors

using StaticArrays

@testset "Laplace" begin
    # Default stencils (4th order)
    operator_path = sbp_operators_path()*"standard_diagonal.toml"
    stencil_set = read_stencil_set(operator_path; order=4)
    g_1D = equidistant_grid(0.0, 1., 101)
    g_3D = equidistant_grid((0.0, -1.0, 0.0), (1., 1., 1.), 51, 101, 52)

    @testset "Constructors" begin
        @testset "1D" begin
            @test Laplace(g_1D, stencil_set) == Laplace(laplace(g_1D, stencil_set), stencil_set)
            @test Laplace(g_1D, stencil_set) isa LazyTensor{Float64,1,1}
        end
        @testset "3D" begin
            @test Laplace(g_3D, stencil_set) == Laplace(laplace(g_3D, stencil_set),stencil_set)
            @test Laplace(g_3D, stencil_set) isa LazyTensor{Float64,3,3}
        end
    end

    # Exact differentiation is measured point-wise. In other cases
    # the error is measured in the l2-norm.
    @testset "Accuracy" begin
        l2(v) = sqrt(prod(spacing.(g_3D.grids))*sum(v.^2));
        polynomials = ()
        maxOrder = 4;
        for i = 0:maxOrder-1
            f_i(x,y,z) = 1/factorial(i)*(y^i + x^i + z^i)
            polynomials = (polynomials...,eval_on(g_3D,f_i))
        end
        # v = eval_on(g_3D, (x,y,z) -> sin(x) + cos(y) + exp(z))
        # Δv = eval_on(g_3D,(x,y,z) -> -sin(x) - cos(y) + exp(z))

        v =  eval_on(g_3D, x̄ -> sin(x̄[1]) + cos(x̄[2]) + exp(x̄[3]))
        Δv = eval_on(g_3D, x̄ -> -sin(x̄[1]) - cos(x̄[2]) + exp(x̄[3]))
        @inferred v[1,2,3]

        # 2nd order interior stencil, 1st order boundary stencil,
        # implies that L*v should be exact for binomials up to order 2.
        @testset "2nd order" begin
            stencil_set = read_stencil_set(operator_path; order=2)
            Δ = Laplace(g_3D, stencil_set)
            @test Δ*polynomials[1] ≈ zeros(Float64, size(g_3D)...) atol = 5e-9
            @test Δ*polynomials[2] ≈ zeros(Float64, size(g_3D)...) atol = 5e-9
            @test Δ*polynomials[3] ≈ polynomials[1] atol = 5e-9
            @test Δ*v ≈ Δv rtol = 5e-2 norm = l2
        end

        # 4th order interior stencil, 2nd order boundary stencil,
        # implies that L*v should be exact for binomials up to order 3.
        @testset "4th order" begin
            stencil_set = read_stencil_set(operator_path; order=4)
            Δ = Laplace(g_3D, stencil_set)
            # NOTE: high tolerances for checking the "exact" differentiation
            # due to accumulation of round-off errors/cancellation errors?
            @test Δ*polynomials[1] ≈ zeros(Float64, size(g_3D)...) atol = 5e-9
            @test Δ*polynomials[2] ≈ zeros(Float64, size(g_3D)...) atol = 5e-9
            @test Δ*polynomials[3] ≈ polynomials[1] atol = 5e-9
            @test Δ*polynomials[4] ≈ polynomials[2] atol = 5e-9
            @test Δ*v ≈ Δv rtol = 5e-4 norm = l2
        end
    end
end

@testset "laplace" begin
    operator_path = sbp_operators_path()*"standard_diagonal.toml"
    stencil_set = read_stencil_set(operator_path; order=4)
    g_1D = equidistant_grid(0.0, 1., 101)
    g_3D = equidistant_grid((0.0, -1.0, 0.0), (1., 1., 1.), 51, 101, 52)

    @testset "EquidistantGrid" begin
        Δ = laplace(g_1D, stencil_set)
        @test Δ == second_derivative(g_1D, stencil_set)
        @test Δ isa LazyTensor{Float64,1,1}
    end
    @testset "TensorGrid" begin
        Δ = laplace(g_3D, stencil_set)
        @test Δ isa LazyTensor{Float64,3,3}
        Dxx = second_derivative(g_3D, stencil_set, 1)
        Dyy = second_derivative(g_3D, stencil_set, 2)
        Dzz = second_derivative(g_3D, stencil_set, 3)
        @test Δ == Dxx + Dyy + Dzz
        @test Δ isa LazyTensor{Float64,3,3}
    end

    @testset "MappedGrid" begin
        c = Chart(unitsquare()) do (ξ,η)
            @SVector[2ξ + η*(1-η), 3η+(1+η/2)*ξ^2]
        end
        Grids.jacobian(c::typeof(c), (ξ,η)) = @SMatrix[2 1-2η; (2+η)*ξ 3+ξ^2/2]

        g = equidistant_grid(c, 60,60)

        @test laplace(g, stencil_set) isa LazyTensor{<:Any,2,2}

        f((x,y)) = sin(4(x + y))
        Δf((x,y)) = -32sin(4(x + y))
        gf = map(f,g)

        Δ = laplace(g, stencil_set)

        @test collect(Δ*gf) isa Array{<:Any,2}
        @test Δ*gf ≈ map(Δf, g) rtol=2e-2
    end
end

@testset "sat_tensors" begin
    # TODO: The following tests should be implemented
    #       1. Symmetry D'H == H'D (test_broken below)
    #       2. Test eigenvalues of and/or solution to Poisson
    #       3. Test tuning of Dirichlet conditions
    #
    #       These tests are likely easiest to implement once
    #       we have support for generating matrices from tensors.

    operator_path = sbp_operators_path()*"standard_diagonal.toml"
    orders = (2,4)
    tols = (5e-2,5e-4)
    sz = (201,401)
    g = equidistant_grid((0.,0.), (1.,1.), sz...)
    
    # Verify implementation of sat_tensors by testing accuracy and symmetry (TODO) 
    # of the operator D = Δ + SAT, where SAT is the tensor composition of the 
    # operators from sat_tensor. Note that SAT*u should approximate 0 for the 
    # conditions chosen.

    @testset "Dirichlet" begin
        for (o, tol) ∈ zip(orders,tols)
            stencil_set = read_stencil_set(operator_path; order=o)
            Δ = Laplace(g, stencil_set)
            H = inner_product(g, stencil_set)
            u = collect(eval_on(g, (x,y) -> sin(π*x)sin(2*π*y)))
            Δu = collect(eval_on(g, (x,y) -> -5*π^2*sin(π*x)sin(2*π*y)))
            D = Δ 
            for id ∈ boundary_identifiers(g)
                D = D + foldl(∘, sat_tensors(Δ, g, DirichletCondition(0., id)))
            end
            e = D*u .- Δu
            # Accuracy
            @test sqrt(sum(H*e.^2)) ≈ 0 atol = tol
            # Symmetry
            r = randn(size(u))
            @test_broken (D'∘H - H∘D)*r .≈ 0 atol = 1e-13 # TODO: Need to implement apply_transpose for D.
        end
    end

    @testset "Neumann" begin
        @testset "Dirichlet" begin
            for (o, tol) ∈ zip(orders,tols)
                stencil_set = read_stencil_set(operator_path; order=o)
                Δ = Laplace(g, stencil_set)
                H = inner_product(g, stencil_set)
                u = collect(eval_on(g, (x,y) -> cos(π*x)cos(2*π*y)))
                Δu = collect(eval_on(g, (x,y) -> -5*π^2*cos(π*x)cos(2*π*y)))
                D = Δ 
                for id ∈ boundary_identifiers(g)
                    D = D + foldl(∘, sat_tensors(Δ, g, NeumannCondition(0., id)))
                end
                e = D*u .- Δu
                # Accuracy
                @test sqrt(sum(H*e.^2)) ≈ 0 atol = tol
                # Symmetry
                r = randn(size(u))
                @test_broken (D'∘H - H∘D)*r .≈ 0 atol = 1e-13 # TODO: Need to implement apply_transpose for D.
            end
        end
    end
end

