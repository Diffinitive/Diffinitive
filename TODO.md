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

## Reasearch and thinking
 - [ ] Use a trait to indicate if a TensorMapping uses indices with regions.
    The default should be that they do NOT.
        - [ ] What to name this trait? Can we call it IndexStyle but not export it to avoid conflicts with Base.IndexStyle?
 - [ ] Use a trait to indicate that a TensorMapping har the same range and domain?
 - [ ] Rename all the Tensor stuff to just LazyOperator, LazyApplication and so on?
 - [ ] Figure out repeated application of regioned TensorMappings. Maybe an instance of a tensor mapping needs to know the exact size of the range and domain for this to work?
 - [ ] Check how the native julia doc generator works
    - [ ] Check if Vidars design docs fit in there
 - [ ] Formalize how range_size() and domain_size() are supposed to work in TensorMappings where dim(domain) != dim(range) (add tests or document)
 - [ ] Create a macro @lazy which replaces a binary op (+,-) by its lazy equivalent? Would be a neat way to indicate which evaluations are lazy without cluttering/confusing with special characters.
 - [ ] Specificera operatorer i TOML eller något liknande?
 - [x] Redo all Tensor applys to take Vararg instead of tuple of Index?
 - [x] Should there be some kind of collection struct for SBP operators (as TensorOperators), providing easy access to all parts (D2, e, d , -> YES!
 H.. H_gamma etc.)
 - [x] Is "missing" a good value for unknown dimension sizes (of `e*g` for example)

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

