# TODO

## Skämskudde
 - [ ] Ändra namn på variabler och funktioner så att det följer style-guide
 - [ ] Skriv tester

## Coding
 - [ ] Add new Laplace opertor to DiffOps, probably named WaveEqOp(?!!?)
 - [ ] Add 1D operators (D1, D2, e, d ... ) as TensorOperators
 - [ ] Create a struct that bundles the necessary Tensor operators for solving the wave equation.
 - [ ] Add a quick and simple way of running all tests for all subpackages.
 - [ ] Replace getindex hack for flatteing tuples with flatten_tuple. (eg. `getindex.(range_size.(L.D2),1)`)
 - [ ] Use `@inferred` in a lot of tests.
 - [ ] Make sure we are setting tolerances in tests in a consistent way
 - [ ] Add check for correct domain sizes to lazy tensor operations using SizeMismatch
 - [ ] Write down some coding guideline or checklist for code convetions. For example i,j,... for indecies and I for multi-index
 - [ ] Add boundschecking in TensorMappingApplication
 - [ ] Start renaming things in LazyTensors
 - [ ] Clean up RegionIndices
    1. [ ] Write tests for how things should work
    2. [ ] Update RegionIndices accordingly
    3. [ ] Fix the rest of the library
 - [ ] Add posibility to create tensor mapping application with `()`, e.g `D1(v) <=> D1*v`?

## Repo
 - [ ] Add Vidar to the authors list
 - [ ] Rename repo to Sbplib.jl

# Wrap up tasks
 - [ ] Kolla att vi har @inbounds och @propagate_inbounds på rätt ställen
 - [ ] Kolla att vi gör boundschecks överallt och att de är markerade med @boundscheck
 - [ ] Kolla att vi har @inline på rätt ställen
 - [ ] Profilera


# Old stuff todos (Are these still relevant?)
Borde det finns motsvarande apply_stencil för apply_quadrature,
apply_boundary_value och apply_normal_derivative?

Borde man alltid skicka in N som parameter i apply_2nd_derivative, t.ex som i
apply_quadrature?

Just nu agerar apply_normal_derivative, apply_boundary_value på inte på v som
en vektor, utan randvärdet plockas ut utanför. Känns inte konsistent med övrig design.
