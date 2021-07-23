using Test

using Sbplib.LazyTensors
using Sbplib.SbpOperators
import Sbplib.SbpOperators: ConstantInteriorScalingOperator

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

    @test range_size(a) == (7,)
    @test domain_size(a) == (7,)

    @test_throws DomainError ConstantInteriorScalingOperator(4,(2,3), 3)
end
