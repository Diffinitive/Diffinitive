# TODO

## Skämskudde
 - [ ] Ändra namn på variabler och funktioner så att det följer style-guide
 - [ ] Skriv tester

## Coding
 - [ ] Add new Laplace opertor to DiffOps, probably named WaveEqOp(?!!?)
 - [ ] Add 1D operators (D1, D2, e, d ... ) as TensorOperators
 - [ ] Create a struct that bundles the necessary Tensor operators for solving the wave equation.
 - [ ] Use traits like IndexStyle, IndexLinear, IndexCartesian to differentiate
    TensorMappings that are flexible in size and those that are fixed in size
 - [ ] Use traits for symmetric tensor mappings such that apply_transpoe = apply for all such mappings
 - [x] Move Laplace tensor operator to different package
 - [x] Remove grid as a property of the Laplace tensor operator
 - [ ] Update how dependencies are handled for tests. This was updated in Julia v1.2 and would allow us to use test specific dev packages.

# Wrap up task

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
en vektor, utan randvärdet plockas ut utanför. Känns inte konsistent med övrig
design

