using SbpOperators
using Test

@testset "apply_quadrature" begin
    op = readOperator(sbp_operators_path()*"d2_4th.txt",sbp_operators_path()*"h_4th.txt")
    h = 0.5

    @test apply_quadrature(op, h, 1.0, 10, 100) == h

    N = 10
    qc = op.quadratureClosure
    q = h.*(qc..., ones(N-2*closuresize(op))..., reverse(qc)...)
    @assert length(q) == N

    for i ∈ 1:N
        @test apply_quadrature(op, h, 1.0, i, N) == q[i]
    end

    v = [2.,3.,2.,4.,5.,4.,3.,4.,5.,4.5]
    for i ∈ 1:N
        @test apply_quadrature(op, h, v[i], i, N) == q[i]*v[i]
    end
end
