# Manifolds, Charts, and Atlases

To construct grids on more complicated geometries we use manifolds described
by one or more charts. The charts describe a mapping from a logical parameter
space to the geometry that we are interested in. If there are more than one
chart for a given geometry this collection of charts and how they are
connected is described by an atlas.

We consider a mapping from the logical coordidinates ``\xi \in \Xi`` to the
physical coordinates ``x \in \Omega``. A `Chart` describes the mapping by a
`ParameterSpace` respresenting ``\Xi`` and some mapping object that takes
arguments ``\xi \in \Xi`` and returns coordinates ``x\in\Omega``. The mapping
object can either be a function or some other callable object.

For the construction of differential and difference operators on a manifold
with a chart the library needs to know the Jacobian,
``\frac{\partial x}{\partial \xi}``, of the mapping as a function of
coordinates in the logical parameter space. Internally, Diffinitive.jl uses a
local Jacobian function, `Grids.jacobian(f, ξ)`. For geometry objects provided
by the library this function should have fast and efficient implementations.
If you are creating your own mapping functions you must implement
`Grids.jacobian` for your function or type, for example

```julia
f(x) = 2x
Grids.jacobian(::typeof(f), x) = fill(2, length(x))
```

```julia
struct F end
(::F)(x) = 2x
Grids.jacobian(::F, x) = fill(2,length(x))
```

You can also let an automatic differentiation tool provide the Jacobian using `with_jacobian`, for example

```julia
using Diffinitive.Grids
using ForwardDiff
using StaticArrays

c = with_jacobian(ForwardDiff.jacobian) do ξ
    @SVector[ξ[1]^2, ξ[1] + ξ[2]]
end
```

Behind the scenes, `with_jacobian` wraps the mapping in a `FunctionWithJacobian` struct. The wrapper forwards calls to the original mapping
and provides a matching `Grids.jacobian` method for that one wrapped object.

`with_jacobian` can also create a `Chart` directly when given a mapping, a
parameter space, and a Jacobian function:

```julia
c = with_jacobian(unitsquare(), ForwardDiff.jacobian) do ξ
    @SVector[ξ[1]^2, ξ[1] + ξ[2]]
end
```

This is equivalent to wrapping the mapping with `with_jacobian` and then
passing the wrapped mapping and parameter space to `Chart`.
