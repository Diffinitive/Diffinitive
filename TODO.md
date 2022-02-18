# TODO


 - [ ] Ändra namn på variabler och funktioner så att det följer style-guide
 - [ ] Add new Laplace operator to DiffOps, probably named WaveEqOp(?!!?)
 - [ ] Create a struct that bundles the necessary Tensor operators for solving the wave equation.
 - [ ] Replace getindex hack for flattening tuples with flatten_tuple. (eg. `getindex.(range_size.(L.D2),1)`)
 - [ ] Use `@inferred` in a lot of tests.
 - [ ] Replace `@inferred` tests with a benchmark suite that automatically tests for regressions.
 - [ ] Make sure we are setting tolerances in tests in a consistent way
 - [ ] Add check for correct domain sizes to lazy tensor operations using SizeMismatch
 - [ ] Write down some coding guideline or checklist for code conventions. For example i,j,... for indices and I for multi-index
 - [ ] Add boundschecking in TensorMappingApplication
 - [ ] Start renaming things in LazyTensors
 - [ ] Clean up RegionIndices
    1. [ ] Write tests for how things should work
    2. [ ] Update RegionIndices accordingly
    3. [ ] Fix the rest of the library
    Should getregion also work for getregion(::Colon,...)
 - [ ] Add possibility to create tensor mapping application with `()`, e.g `D1(v) <=> D1*v`?
 - [ ] Add custom pretty printing to LazyTensors/SbpOperators to enhance readability of e.g error messages.
       See (https://docs.julialang.org/en/v1/manual/types/#man-custom-pretty-printing)
 - [ ] Samla noggrannhets- och SBP-ness-tester för alla operatorer på ett ställe


 - [ ] Gå igenom alla typ parametrar och kolla om de är motiverade. Både i signaturer och typer, tex D i VariableSecondDerivative. Kan vi använda promote istället?
 - [ ] Kolla att vi har @inbounds och @propagate_inbounds på rätt ställen
 - [ ] Kolla att vi gör boundschecks överallt och att de är markerade med @boundscheck
 - [ ] Kolla att vi har @inline på rätt ställen
 - [ ] Profilera
