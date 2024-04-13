using Test

using Sbplib.LazyTensors
using Sbplib.SbpOperators
import Sbplib.SbpOperators: ConstantInteriorScalingOperator
using Sbplib.Grids

@testset "ConstantInteriorScalingOperator" begin
    @test ConstantInteriorScalingOperator(1, (2,3), 10) isa ConstantInteriorScalingOperator{Int,2}
    @test ConstantInteriorScalingOperator(1., (2.,3.), 10) isa ConstantInteriorScalingOperator{Float64,2}

    a = ConstantInteriorScalingOperator(4, (2,3), 10)
    v = ones(Int, 10)
    @test a*v == [2,3,4,4,4,4,4,4,3,2]
    @test a'*v == [2,3,4,4,4,4,4,4,3,2]

    @test range_size(a) == (10,)
    @test domain_size(a) == (10,)


    a = ConstantInteriorScalingOperator(.5, (.1,.2), 7)
    v = ones(7)

    @test a*v == [.1,.2,.5,.5,.5,.2,.1]
    @test a'*v == [.1,.2,.5,.5,.5,.2,.1]

    @test (a*rand(ComplexF64, domain_size(a)... ))[1] isa ComplexF64
    @test (a'*rand(ComplexF64, domain_size(a')...))[1] isa ComplexF64

    @test range_size(a) == (7,)
    @test domain_size(a) == (7,)

    @test_throws DomainError ConstantInteriorScalingOperator(4,(2,3), 3)

    @testset "Grid constructor" begin
        g = equidistant_grid(0., 2., 11)
        @test ConstantInteriorScalingOperator(g, 3., (.1,.2)) isa ConstantInteriorScalingOperator{Float64}
    end
end
