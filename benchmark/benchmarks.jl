using BenchmarkTools
using Sbplib
using Sbplib.Grids
using Sbplib.SbpOperators

const SUITE = BenchmarkGroup()


sz(d) = ntuple(i->100, d)
ll(d) = ntuple(i->0., d)
lu(d) = ntuple(i->1., d)

g1 = EquidistantGrid(sz(1),ll(1),lu(1))
g2 = EquidistantGrid(sz(2),ll(2),lu(2))
g3 = EquidistantGrid(sz(3),ll(3),lu(3))

v1 = rand(sz(1)...)
v2 = rand(sz(2)...)
v3 = rand(sz(3)...)

u1 = rand(sz(1)...)
u2 = rand(sz(2)...)
u3 = rand(sz(3)...)

stencil_set = read_stencil_set(joinpath(sbp_operators_path(),"standard_diagonal.toml"); order=4)

SUITE["derivatives"] = BenchmarkGroup()


SUITE["derivatives"]["first_derivative"] = BenchmarkGroup()

D₁ = first_derivative(g1,stencil_set)
SUITE["derivatives"]["first_derivative"]["1D"] = @benchmarkable $u1 .= $D₁*$v1

Dx = first_derivative(g2,stencil_set,1)
Dy = first_derivative(g2,stencil_set,2)
SUITE["derivatives"]["first_derivative"]["2D"] = BenchmarkGroup()
SUITE["derivatives"]["first_derivative"]["2D"]["x"] = @benchmarkable $u2 .= $Dx*$v2
SUITE["derivatives"]["first_derivative"]["2D"]["y"] = @benchmarkable $u2 .= $Dy*$v2

Dx = first_derivative(g3,stencil_set,1)
Dy = first_derivative(g3,stencil_set,2)
Dz = first_derivative(g3,stencil_set,3)
SUITE["derivatives"]["first_derivative"]["3D"] = BenchmarkGroup()
SUITE["derivatives"]["first_derivative"]["3D"]["x"] = @benchmarkable $u3 .= $Dx*$v3
SUITE["derivatives"]["first_derivative"]["3D"]["y"] = @benchmarkable $u3 .= $Dy*$v3
SUITE["derivatives"]["first_derivative"]["3D"]["z"] = @benchmarkable $u3 .= $Dz*$v3


SUITE["derivatives"]["second_derivative"] = BenchmarkGroup()

D₁ = second_derivative(g1,stencil_set)
SUITE["derivatives"]["second_derivative"]["1D"] = @benchmarkable $u1 .= $D₁*$v1

Dx = second_derivative(g2,stencil_set,1)
Dy = second_derivative(g2,stencil_set,2)
SUITE["derivatives"]["second_derivative"]["2D"] = BenchmarkGroup()
SUITE["derivatives"]["second_derivative"]["2D"]["x"] = @benchmarkable $u2 .= $Dx*$v2
SUITE["derivatives"]["second_derivative"]["2D"]["y"] = @benchmarkable $u2 .= $Dy*$v2

Dx = second_derivative(g3,stencil_set,1)
Dy = second_derivative(g3,stencil_set,2)
Dz = second_derivative(g3,stencil_set,3)
SUITE["derivatives"]["second_derivative"]["3D"] = BenchmarkGroup()
SUITE["derivatives"]["second_derivative"]["3D"]["x"] = @benchmarkable $u3 .= $Dx*$v3
SUITE["derivatives"]["second_derivative"]["3D"]["y"] = @benchmarkable $u3 .= $Dy*$v3
SUITE["derivatives"]["second_derivative"]["3D"]["z"] = @benchmarkable $u3 .= $Dz*$v3


SUITE
