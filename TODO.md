# TODO

## Organization
 - [ ] Split up Notes.md in several files

## Coding
 - [ ] Create a struct that bundles the necessary Tensor operators for solving the wave equation.
 - [ ] Make sure we are setting tolerances in tests in a consistent way
 - [ ] Write down some coding guideline or checklist for code conventions. For example i,j,... for indices and I for multi-index
 - [ ] Clean up RegionIndices
    1. [ ] Write tests for how things should work
    2. [ ] Update RegionIndices accordingly
    3. [ ] Fix the rest of the library
    Should getregion also work for getregion(::Colon,...)
 - [ ] Add custom pretty printing to LazyTensors/SbpOperators to enhance readability of e.g error messages.
       See (https://docs.julialang.org/en/v1/manual/types/#man-custom-pretty-printing)
 - [ ] Samla noggrannhets- och SBP-ness-tester för alla operatorer på ett ställe
 - [ ] Move export statements to top of each module
 - [ ] Implement apply_transpose for
      - [ ] ElementwiseTensorOperation
      - [ ] VolumeOperator
      - [ ] Laplace


 - [ ] Gå igenom alla typ parametrar och kolla om de är motiverade. Både i signaturer och typer, tex D i VariableSecondDerivative. Kan vi använda promote istället?
 - [ ] Kolla att vi har @inbounds och @propagate_inbounds på rätt ställen
 - [ ] Kolla att vi gör boundschecks överallt och att de är markerade med @boundscheck
 - [ ] Kolla att vi har @inline på rätt ställen
 - [ ] Profilera

 - [ ] Keep a lookout for allowing dependencies of package extensions (https://github.com/JuliaLang/Pkg.jl/issues/3641) This should be used to simplify the matrix extensions so that you don't have to load Tokens which is only used internally to the extension

### Grids

 - [ ] Multiblock grids
 - [ ] Periodic grids
 - [ ] Grids with modified boundary closures
 - [ ] Support indexing with `:`.


### Benchmarks
 - [ ] Benchmarks for all grid indexing (focused on allocation)
 - [ ] Benchmarks for indexing of lazy grid functions
 - [ ] Add benchmarks for range type in EquidistantGrid. (LinRange vs StepRange)
