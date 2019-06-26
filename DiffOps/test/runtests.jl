using Test
using DiffOps
using Grids
using SbpOperators
using RegionIndices
using LazyTensors

@test_broken false

@testset "BoundaryValue" begin
    op = readOperator(sbp_operators_path()*"d2_4th.txt",sbp_operators_path()*"h_4th.txt")
    g = EquidistantGrid((4,5), (0.0, 0.0), (1.0,1.0))

    e_w = BoundaryValue(op, g, CartesianBoundary{1,Lower}())
    e_e = BoundaryValue(op, g, CartesianBoundary{1,Upper}())
    e_s = BoundaryValue(op, g, CartesianBoundary{2,Lower}())
    e_n = BoundaryValue(op, g, CartesianBoundary{2,Upper}())

    v = zeros(Float64, 4, 5)
    v[:,5] = [1, 2, 3,4]
    v[:,4] = [1, 2, 3,4]
    v[:,3] = [4, 5, 6, 7]
    v[:,2] = [7, 8, 9, 10]
    v[:,1] = [10, 11, 12, 13]

    @test e_w  isa TensorMapping{T,2,1} where T
    @test e_w' isa TensorMapping{T,1,2} where T

    @test domain_size(e_w, (3,2)) == (2,)
    @test domain_size(e_e, (3,2)) == (2,)
    @test domain_size(e_s, (3,2)) == (3,)
    @test domain_size(e_n, (3,2)) == (3,)

    @test size(e_w'*v) == (5,)
    @test size(e_e'*v) == (5,)
    @test size(e_s'*v) == (4,)
    @test size(e_n'*v) == (4,)

    @test collect(e_w'*v) == [10,7,4,1.0,1]
    @test collect(e_e'*v) == [13,10,7,4,4.0]
    @test collect(e_s'*v) == [10,11,12,13.0]
    @test collect(e_n'*v) == [1,2,3,4.0]

end
