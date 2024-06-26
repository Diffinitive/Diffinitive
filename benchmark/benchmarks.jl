using BenchmarkTools

using Sbplib
using Sbplib.Grids
using Sbplib.SbpOperators
using Sbplib.RegionIndices
using Sbplib.LazyTensors

using LinearAlgebra

const SUITE = BenchmarkGroup()


sz(d) = ntuple(i->100, d)
ll(d) = ntuple(i->0., d)
lu(d) = ntuple(i->1., d)

g1 = equidistant_grid(ll(1)[1], lu(1)[1], sz(1)...)
g2 = equidistant_grid(ll(2), lu(2), sz(2)...)
g3 = equidistant_grid(ll(3), lu(3), sz(3)...)

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

D₂ = second_derivative(g1,stencil_set)
SUITE["derivatives"]["second_derivative"]["1D"] = @benchmarkable $u1 .= $D₂*$v1

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


SUITE["derivatives"]["second_derivative_variable"] = BenchmarkGroup()

c1 = map(x->sin(x)+2, g1)
D₂ = second_derivative_variable(g1, c1, stencil_set)
SUITE["derivatives"]["second_derivative_variable"]["1D"] = @benchmarkable $u1 .= $D₂*$v1

c2 = map(x->sin(x[1] + x[2])+2, g2)
Dx = second_derivative_variable(g2, c2, stencil_set, 1)
Dy = second_derivative_variable(g2, c2, stencil_set, 2)
SUITE["derivatives"]["second_derivative_variable"]["2D"] = BenchmarkGroup()
SUITE["derivatives"]["second_derivative_variable"]["2D"]["x"] = @benchmarkable $u2 .= $Dx*$v2
SUITE["derivatives"]["second_derivative_variable"]["2D"]["y"] = @benchmarkable $u2 .= $Dy*$v2

c3 = map(x->sin(norm(x))+2, g3)
Dx = second_derivative_variable(g3, c3, stencil_set, 1)
Dy = second_derivative_variable(g3, c3, stencil_set, 2)
Dz = second_derivative_variable(g3, c3, stencil_set, 3)
SUITE["derivatives"]["second_derivative_variable"]["3D"] = BenchmarkGroup()
SUITE["derivatives"]["second_derivative_variable"]["3D"]["x"] = @benchmarkable $u3 .= $Dx*$v3
SUITE["derivatives"]["second_derivative_variable"]["3D"]["y"] = @benchmarkable $u3 .= $Dy*$v3
SUITE["derivatives"]["second_derivative_variable"]["3D"]["z"] = @benchmarkable $u3 .= $Dz*$v3




SUITE["derivatives"]["addition"] = BenchmarkGroup()

D₁ = first_derivative(g1,stencil_set)
D₂ = second_derivative(g1,stencil_set)
SUITE["derivatives"]["addition"]["1D"] = BenchmarkGroup()
SUITE["derivatives"]["addition"]["1D"]["apply,add"] = @benchmarkable $u1 .= $D₁*$v1 + $D₂*$v1
SUITE["derivatives"]["addition"]["1D"]["add,apply"] = @benchmarkable $u1 .= ($D₁ + $D₂)*$v1

Dxx = second_derivative(g2,stencil_set,1)
Dyy = second_derivative(g2,stencil_set,2)
SUITE["derivatives"]["addition"]["2D"] = BenchmarkGroup()
SUITE["derivatives"]["addition"]["2D"]["apply,add"] = @benchmarkable $u2 .= $Dxx*$v2 + $Dyy*$v2
SUITE["derivatives"]["addition"]["2D"]["add,apply"] = @benchmarkable $u2 .= ($Dxx + $Dyy)*$v2

Dxx = second_derivative(g3,stencil_set,1)
Dyy = second_derivative(g3,stencil_set,2)
Dzz = second_derivative(g3,stencil_set,3)
SUITE["derivatives"]["addition"]["3D"] = BenchmarkGroup()
SUITE["derivatives"]["addition"]["3D"]["apply,add"] = @benchmarkable $u3 .= $Dxx*$v3 + $Dyy*$v3 + $Dzz*$v3
SUITE["derivatives"]["addition"]["3D"]["add,apply"] = @benchmarkable $u3 .= ($Dxx + $Dyy + $Dzz)*$v3


SUITE["derivatives"]["composition"] = BenchmarkGroup()

Dx = first_derivative(g1,stencil_set)
SUITE["derivatives"]["composition"]["1D"] = BenchmarkGroup()
SUITE["derivatives"]["composition"]["1D"]["apply,apply"] = @benchmarkable $u1 .= $Dx*($Dx*$v1)
SUITE["derivatives"]["composition"]["1D"]["compose,apply"] = @benchmarkable $u1 .= ($Dx∘$Dx)*$v1

Dx = first_derivative(g2,stencil_set,1)
Dy = first_derivative(g2,stencil_set,2)
SUITE["derivatives"]["composition"]["2D"] = BenchmarkGroup()
SUITE["derivatives"]["composition"]["2D"]["apply,apply"] = @benchmarkable $u2 .= $Dy*($Dx*$v2)
SUITE["derivatives"]["composition"]["2D"]["compose,apply"] = @benchmarkable $u2 .= ($Dy∘$Dx)*$v2

Dx = first_derivative(g3,stencil_set,1)
Dy = first_derivative(g3,stencil_set,2)
Dz = first_derivative(g3,stencil_set,3)
SUITE["derivatives"]["composition"]["3D"] = BenchmarkGroup()
SUITE["derivatives"]["composition"]["3D"]["xy"] = BenchmarkGroup()
SUITE["derivatives"]["composition"]["3D"]["xy"]["apply,apply"] = @benchmarkable $u3 .= $Dx*($Dy*$v3)
SUITE["derivatives"]["composition"]["3D"]["xy"]["compose,apply"] = @benchmarkable $u3 .= ($Dx∘$Dy)*$v3

SUITE["derivatives"]["composition"]["3D"]["yz"] = BenchmarkGroup()
SUITE["derivatives"]["composition"]["3D"]["yz"]["apply,apply"] = @benchmarkable $u3 .= $Dy*($Dz*$v3)
SUITE["derivatives"]["composition"]["3D"]["yz"]["compose,apply"] = @benchmarkable $u3 .= ($Dy∘$Dz)*$v3

SUITE["derivatives"]["composition"]["3D"]["xz"] = BenchmarkGroup()
SUITE["derivatives"]["composition"]["3D"]["xz"]["apply,apply"] = @benchmarkable $u3 .= $Dx*($Dz*$v3)
SUITE["derivatives"]["composition"]["3D"]["xz"]["compose,apply"] = @benchmarkable $u3 .= ($Dx∘$Dz)*$v3

SUITE["derivatives"]["composition"]["3D"]["xx"] = BenchmarkGroup()
SUITE["derivatives"]["composition"]["3D"]["xx"]["apply,apply"] = @benchmarkable $u3 .= $Dx*($Dx*$v3)
SUITE["derivatives"]["composition"]["3D"]["xx"]["compose,apply"] = @benchmarkable $u3 .= ($Dx∘$Dx)*$v3

SUITE["derivatives"]["composition"]["3D"]["yy"] = BenchmarkGroup()
SUITE["derivatives"]["composition"]["3D"]["yy"]["apply,apply"] = @benchmarkable $u3 .= $Dy*($Dy*$v3)
SUITE["derivatives"]["composition"]["3D"]["yy"]["compose,apply"] = @benchmarkable $u3 .= ($Dy∘$Dy)*$v3

SUITE["derivatives"]["composition"]["3D"]["zz"] = BenchmarkGroup()
SUITE["derivatives"]["composition"]["3D"]["zz"]["apply,apply"] = @benchmarkable $u3 .= $Dz*($Dz*$v3)
SUITE["derivatives"]["composition"]["3D"]["zz"]["compose,apply"] = @benchmarkable $u3 .= ($Dz∘$Dz)*$v3


SUITE["boundary_terms"] = BenchmarkGroup()

H = inner_product(g2, stencil_set)
H⁻¹ = inverse_inner_product(g2, stencil_set)
Dxx = second_derivative(g2, stencil_set, 1)
Dyy = second_derivative(g2, stencil_set, 2)

e₁ₗ = boundary_restriction(g2, stencil_set, CartesianBoundary{1,Lower}())
e₁ᵤ = boundary_restriction(g2, stencil_set, CartesianBoundary{1,Upper}())
e₂ₗ = boundary_restriction(g2, stencil_set, CartesianBoundary{2,Lower}())
e₂ᵤ = boundary_restriction(g2, stencil_set, CartesianBoundary{2,Upper}())

d₁ₗ = normal_derivative(g2, stencil_set, CartesianBoundary{1,Lower}())
d₁ᵤ = normal_derivative(g2, stencil_set, CartesianBoundary{1,Upper}())
d₂ₗ = normal_derivative(g2, stencil_set, CartesianBoundary{2,Lower}())
d₂ᵤ = normal_derivative(g2, stencil_set, CartesianBoundary{2,Upper}())

H₁ₗ = inner_product(boundary_grid(g2, CartesianBoundary{1,Lower}()), stencil_set)
H₁ᵤ = inner_product(boundary_grid(g2, CartesianBoundary{1,Upper}()), stencil_set)
H₂ₗ = inner_product(boundary_grid(g2, CartesianBoundary{2,Lower}()), stencil_set)
H₂ᵤ = inner_product(boundary_grid(g2, CartesianBoundary{2,Upper}()), stencil_set)

SUITE["boundary_terms"]["pre_composition"] = @benchmarkable $u2 .= $(H⁻¹∘e₁ₗ'∘H₁ₗ∘d₁ₗ)*$v2
SUITE["boundary_terms"]["composition"]     = @benchmarkable $u2 .= ($H⁻¹∘$e₁ₗ'∘$H₁ₗ∘$d₁ₗ)*$v2
SUITE["boundary_terms"]["application"]     = @benchmarkable $u2 .= $H⁻¹*$e₁ₗ'*$H₁ₗ* $d₁ₗ*$v2
# An investigation of these allocations can be found in the branch `allocation_test`

#TODO: Reorg with dimension as first level? To reduce operator creation?



SUITE["lazy_tensors"] = BenchmarkGroup()

SUITE["lazy_tensors"]["compositions"] = BenchmarkGroup()
s = ScalingTensor(1.,(10,))
u = rand(10)
v = similar(u)
s3 = s∘s∘s
s4 = s∘s∘s∘s
SUITE["lazy_tensors"]["compositions"]["s∘s∘s"]   = @benchmarkable $v .= $s3*$u
SUITE["lazy_tensors"]["compositions"]["s∘s∘s∘s"] = @benchmarkable $v .= $s4*$u


SUITE
