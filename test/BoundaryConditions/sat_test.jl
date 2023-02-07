using Test


using Sbplib.BoundaryConditions
using Sbplib.Grids
using Sbplib.RegionIndices
using Sbplib.LazyTensors

grid = EquidistantGrid(11, 0.0, 1.0)
(id_l,id_r) = boundary_identifiers(grid)
struct MockOp
end

function BoundaryConditions.sat_tensors(op::MockOp, grid, bc::DirichletCondition)
    sz = size(grid)
    m = sz[1]
    ind = (region(bc.id) == Lower()) ? 1 : m
    e = zeros(m);
    e[ind] = 1.
    eᵀ = ones(Float64,m,0);
    e[ind] = 1.
    c_tensor = LazyTensors.DiagonalTensor(e)
    p_tensor = DenseTensor(eᵀ, (1,), (2,))
    closure(u) = c_tensor*u
    function penalty(g)
        @show g
        return p_tensor*g
    end
    return closure, penalty
end


@testset "sat" begin
    g = ConstantBoundaryData(2.0)
    dc = DirichletCondition(g,id_l)
    op = MockOp()
    f = sat(op, grid, dc)
    u = evalOn(grid, x-> -1/2 + x^2)
    @show f(0.,u)
end
