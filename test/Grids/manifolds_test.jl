using Test

using Sbplib.Grids
using Sbplib.RegionIndices
using Sbplib.LazyTensors

# using StaticArrays

@testset "HyperBox" begin
    @test HyperBox([1,1], [2,2]) isa HyperBox{Int, 2}

    @test limits(HyperBox([1,2], [3,4])) == ([1,2], [3,4])
    @test limits(HyperBox([1,2], [3,4]), 1) == (1,3)
    @test limits(HyperBox([1,2], [3,4]), 2) == (2,4)

    @test unitinterval() isa HyperBox{Float64,1}
    @test limits(unitinterval()) == ([0], [1])

    @test unitinterval(Int) isa HyperBox{Int,1}
    @test limits(unitinterval(Int)) == ([0], [1])

    @test unitsquare() isa HyperBox{Float64,2}
    @test limits(unitsquare()) == ([0,0],[1,1])

    @test unitcube() isa HyperBox{Float64,3}
    @test limits(unitcube()) == ([0,0,0],[1,1,1])

    @test unithyperbox(4) isa HyperBox{Float64,4}
    @test limits(unithyperbox(4)) == ([0,0,0,0],[1,1,1,1])
end

@testset "Simplex" begin
    @test Simplex([1,2], [3,4]) isa Simplex{Int, 2}
    @test Simplex([1,2,3], [4,5,6],[1,1,1]) isa Simplex{Int, 3}
end

@testset "Chart" begin
end

@testset "ConcreteChart" begin
    c = ConcreteChart(x->2x, unitsquare())
    @test c isa Chart{2}
    @test c([3,2]) == [6,4]
    @test parameterspace(c) == unitsquare()
end

@testset "Atlas" begin

end
