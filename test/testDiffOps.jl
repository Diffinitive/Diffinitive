using Test
using Sbplib.DiffOps
using Sbplib.Grids
using Sbplib.SbpOperators
using Sbplib.RegionIndices
using Sbplib.LazyTensors

@testset "DiffOps" begin
#
# @testset "BoundaryValue" begin
#     op = read_D2_operator(sbp_operators_path()*"standard_diagonal.toml"; order=4)
#     g = EquidistantGrid((4,5), (0.0, 0.0), (1.0,1.0))
#
#     e_w = BoundaryValue(op, g, CartesianBoundary{1,Lower}())
#     e_e = BoundaryValue(op, g, CartesianBoundary{1,Upper}())
#     e_s = BoundaryValue(op, g, CartesianBoundary{2,Lower}())
#     e_n = BoundaryValue(op, g, CartesianBoundary{2,Upper}())
#
#     v = zeros(Float64, 4, 5)
#     v[:,5] = [1, 2, 3,4]
#     v[:,4] = [1, 2, 3,4]
#     v[:,3] = [4, 5, 6, 7]
#     v[:,2] = [7, 8, 9, 10]
#     v[:,1] = [10, 11, 12, 13]
#
#     @test e_w  isa TensorMapping{T,2,1} where T
#     @test e_w' isa TensorMapping{T,1,2} where T
#
#     @test domain_size(e_w, (3,2)) == (2,)
#     @test domain_size(e_e, (3,2)) == (2,)
#     @test domain_size(e_s, (3,2)) == (3,)
#     @test domain_size(e_n, (3,2)) == (3,)
#
#     @test size(e_w'*v) == (5,)
#     @test size(e_e'*v) == (5,)
#     @test size(e_s'*v) == (4,)
#     @test size(e_n'*v) == (4,)
#
#     @test collect(e_w'*v) == [10,7,4,1.0,1]
#     @test collect(e_e'*v) == [13,10,7,4,4.0]
#     @test collect(e_s'*v) == [10,11,12,13.0]
#     @test collect(e_n'*v) == [1,2,3,4.0]
#
#     g_x = [1,2,3,4.0]
#     g_y = [5,4,3,2,1.0]
#
#     G_w = zeros(Float64, (4,5))
#     G_w[1,:] = g_y
#
#     G_e = zeros(Float64, (4,5))
#     G_e[4,:] = g_y
#
#     G_s = zeros(Float64, (4,5))
#     G_s[:,1] = g_x
#
#     G_n = zeros(Float64, (4,5))
#     G_n[:,5] = g_x
#
#     @test size(e_w*g_y) == (UnknownDim,5)
#     @test size(e_e*g_y) == (UnknownDim,5)
#     @test size(e_s*g_x) == (4,UnknownDim)
#     @test size(e_n*g_x) == (4,UnknownDim)
#
#     # These tests should be moved to where they are possible (i.e we know what the grid should be)
#     @test_broken collect(e_w*g_y) == G_w
#     @test_broken collect(e_e*g_y) == G_e
#     @test_broken collect(e_s*g_x) == G_s
#     @test_broken collect(e_n*g_x) == G_n
# end
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
#     @test collect(d_w'*v) ≈ v∂x[1,:]
#     @test collect(d_e'*v) ≈ v∂x[5,:]
#     @test collect(d_s'*v) ≈ v∂y[:,1]
#     @test collect(d_n'*v) ≈ v∂y[:,6]
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
#     @test_broken collect(d_w*g_y) ≈ G_w
#     @test_broken collect(d_e*g_y) ≈ G_e
#     @test_broken collect(d_s*g_x) ≈ G_s
#     @test_broken collect(d_n*g_x) ≈ G_n
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
#     @test collect(H_w*v_w) ≈ q_y.*v_w
#     @test collect(H_e*v_e) ≈ q_y.*v_e
#     @test collect(H_s*v_s) ≈ q_x.*v_s
#     @test collect(H_n*v_n) ≈ q_x.*v_n
#
#     @test collect(H_w'*v_w) == collect(H_w'*v_w)
#     @test collect(H_e'*v_e) == collect(H_e'*v_e)
#     @test collect(H_s'*v_s) == collect(H_s'*v_s)
#     @test collect(H_n'*v_n) == collect(H_n'*v_n)
# end

end
