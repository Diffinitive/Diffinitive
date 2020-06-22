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
 - [x] Move Laplace tensor operator to different package
 - [x] Remove grid as a property of the Laplace tensor operator

## Reasearch and thinking
 - [ ] Redo all Tensor applys to take Vararg instead of tuple of Index?
    Have we been down that road before? Is there any reason not to do this?
 - [ ] Check how the native julia doc generator works
    - [ ] Check if Vidars design docs fit in there
 - [ ] Formalize how range_size() and domain_size() are supposed to work in TensorMappings where dim(domain) != dim(range) (add tests or document)
 - [x] Should there be some kind of collection struct for SBP operators (as TensorOperators), providing easy access to all parts (D2, e, d , -> YES!
 H.. H_gamma etc.)
 - [x] Is "missing" a good value for unknown dimension sizes (of `e*g` for example)
 - [] Add traits for symmetric tensor mappings such that apply_transpoe = apply for all such mappings

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

Specificera operatorer i TOML eller något liknande?
