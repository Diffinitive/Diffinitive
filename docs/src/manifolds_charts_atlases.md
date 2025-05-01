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

You can also provide a fallback function using one of the many automatic
differentiation packages, for example

```julia
using ForwardDiff
Grids.jacobian(f,x) = ForwardDiff.jacobian(f,x)
```
