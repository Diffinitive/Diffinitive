# LazyTensors and operations

`LazyTensor`s are matrix-free representations of linear operators acting on
multi-dimensional arrays. A `LazyTensor{R,D}` maps a `D`-dimensional array to an
`R`-dimensional array. Instead of storing all tensor entries, each concrete type
implements how to compute one output entry from an input array.

The core interface consists of:

* [`LazyTensor`](@ref), the abstract operator type.
* [`range_size`](@ref) and [`domain_size`](@ref), which describe the output and
  input array sizes.
* [`apply`](@ref), which computes one output value.
* [`apply_transpose`](@ref), which is optional and is used by transposed
  tensors.

This is the representation used by operators in `SbpOperators`. For example, a
first derivative operator is a `LazyTensor`, not a stored differentiation
matrix.

```jldoctest lazy_tensors; output = false
using Diffinitive.Grids
using Diffinitive.LazyTensors
using Diffinitive.SbpOperators

stencil_set = read_stencil_set(sbp_operators_path() * "standard_diagonal.toml"; order = 2)
g = equidistant_grid(0.0, 1.0, 11)
D = first_derivative(g, stencil_set)

# output

Diffinitive.SbpOperators.VolumeOperator{Float64, 3, 1, 2}(Stencil{Float64, 3}(-1:1, (-5.0, 0.0, 5.0)), (Stencil{Float64, 2}(0:1, (-10.0, 10.0)),), 11, Diffinitive.SbpOperators.odd)
```

```jldoctest lazy_tensors
julia> D isa LazyTensor{1,1}
true

julia> domain_size(D)
(11,)

julia> range_size(D)
(11,)
```

## Tensor applications

Applying a `LazyTensor` to an array with `*` creates a
[`TensorApplication`](@ref). A `TensorApplication` behaves like an
`AbstractArray`, but the entries are computed when they are indexed.

```jldoctest lazy_tensors; output = false
u = map(x -> x^2, g)
du = D * u

# output

11-element TensorApplication{Float64, 1, 1, Diffinitive.SbpOperators.VolumeOperator{Float64, 3, 1, 2}, Vector{Float64}}:
 0.10000000000000002
 0.20000000000000004
 0.39999999999999997
 0.6000000000000001
 0.8
 0.9999999999999997
 1.1999999999999997
 1.4000000000000008
 1.600000000000001
 1.7999999999999994
 1.8999999999999986
```

Now `u` is a regular array
```jldoctest lazy_tensors
julia> u
11-element Vector{Float64}:
 0.0
 0.010000000000000002
 0.04000000000000001
 0.09
 0.16000000000000003
 0.25
 0.36
 0.48999999999999994
 0.6400000000000001
 0.81
 1.0
```
and `du` is an `AbstractArray` which computes elements on indexing
```jldoctest lazy_tensors
julia> du
11-element TensorApplication{Float64, 1, 1, Diffinitive.SbpOperators.VolumeOperator{Float64, 3, 1, 2}, Vector{Float64}}:
 0.10000000000000002
 0.20000000000000004
 0.39999999999999997
 0.6000000000000001
 0.8
 0.9999999999999997
 1.1999999999999997
 1.4000000000000008
 1.600000000000001
 1.7999999999999994
 1.8999999999999986
```

The full result can be materialized with `collect`, but this is not required for
ordinary array operations. For this example, the derivative of `x^2` at
`x = 0.5` is `1.0`, so the indexed value agrees with the exact derivative at the
center grid point:

```jldoctest lazy_tensors
julia> du[6] ≈ 2g[6]
true
```

The same mechanism is used for tensor-product grids. In more than one dimension,
the range and domain sizes are tuples with one entry per grid direction.

```jldoctest lazy_tensors; output = false
g2 = equidistant_grid((0.0, 0.0), (1.0, 1.0), 11, 12)
Dx = first_derivative(g2, stencil_set, 1)
Dy = first_derivative(g2, stencil_set, 2)
v = map(x -> x[1] + 2x[2], g2)

# output

11×12 Matrix{Float64}:
 0.0  0.181818  0.363636  0.545455  …  1.45455  1.63636  1.81818  2.0
 0.1  0.281818  0.463636  0.645455     1.55455  1.73636  1.91818  2.1
 0.2  0.381818  0.563636  0.745455     1.65455  1.83636  2.01818  2.2
 0.3  0.481818  0.663636  0.845455     1.75455  1.93636  2.11818  2.3
 0.4  0.581818  0.763636  0.945455     1.85455  2.03636  2.21818  2.4
 0.5  0.681818  0.863636  1.04545   …  1.95455  2.13636  2.31818  2.5
 0.6  0.781818  0.963636  1.14545      2.05455  2.23636  2.41818  2.6
 0.7  0.881818  1.06364   1.24545      2.15455  2.33636  2.51818  2.7
 0.8  0.981818  1.16364   1.34545      2.25455  2.43636  2.61818  2.8
 0.9  1.08182   1.26364   1.44545      2.35455  2.53636  2.71818  2.9
 1.0  1.18182   1.36364   1.54545   …  2.45455  2.63636  2.81818  3.0
```

```jldoctest lazy_tensors
julia> domain_size(Dx)
(11, 12)

julia> range_size(Dy)
(11, 12)

julia> (Dx * v)[6, 6]
1.0

julia> (Dy * v)[6, 6]
2.0
```

## Lazy operations

Operations on `LazyTensor`s create new `LazyTensor`s. The computation is still
deferred until the resulting tensor is applied to an array and indexed.

Available operations include:

* `+` for lazy sums.
* `-` for lazy negation and subtraction.
* scalar multiplication with `*`.
* `∘` for lazy composition.
* `'` for lazy transposition, when the underlying tensor implements
  [`apply_transpose`](@ref).
* `⊗` for lazy outer products.

For example, `Dx + Dy` creates a lazy sum of two derivative operators:

```jldoctest lazy_tensors
julia> Dx+Dy isa LazyTensor{2,2}
true

julia> ((Dx+Dy) * v)[6, 6]
3.0
```

Scalar multiplication builds a lazy scaling tensor around the original tensor:

```jldoctest lazy_tensors
julia> (3 * Dx * v)[6, 6]
3.0
```

Composition uses Julia's function-composition operator. If `A` and `B` are lazy
tensors, then `A ∘ B` represents applying `B` first and then `A`.

```jldoctest lazy_tensors
julia> Dxy = Dy ∘ Dy;

julia> (Dxy * v)[6,6]
0.0
```

Transposition is also lazy:

```jldoctest lazy_tensors
julia> D' isa LazyTensor{1,1}
true
```

## Simplifications

The operator overloads perform a few basic simplifications. These are intended
to keep common expressions compact while preserving the lazy representation.

For example, adding a compatible zero tensor returns the original tensor:

```jldoctest lazy_tensors
julia> D + zero(D) === D
true
```

Composition with compatible identity tensors removes the identity:

```jldoctest lazy_tensors
julia> I = IdentityTensor(domain_size(D));

julia> D ∘ I === D
true
```

Nested sums are flattened into a single [`TensorSum`](@ref), which keeps
expressions like `Dx + Dy + Dx` from building deeply nested binary trees:

These simplifications are deliberately small. They check sizes, but they do not
try to prove algebraic identities such as `D - D == 0`.

## Constructors for explicit expression trees

The overloaded operations are convenient and may simplify the expression. When
you need full control over the exact lazy operation object, use the constructor
types directly:

* [`TensorApplication`](@ref) for applying a tensor to an array.
* [`TensorSum`](@ref) for a lazy sum.
* [`TensorNegation`](@ref) for a lazy negation.
* [`TensorComposition`](@ref) for a lazy composition.
* [`TensorTranspose`](@ref) for a lazy transpose.

These constructors perform the relevant size checks, but they do not apply the
extra simplifications provided by the overloaded operations.

Direct constructors are mostly useful in generated code, tests, and situations
where the shape of the lazy expression is itself important. In ordinary user
code, prefer the operators because they are shorter and include the common
simplifications.
