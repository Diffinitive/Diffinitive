# Notes

## Known size of range and domain?
Is there any reason to use a trait to differentiate between fixed size and unknown size?

When do we need to know the size of the range and domain?
 * When indexing to provide boundschecking?
 * When doing specialised computations for different parts of the range/domain?
 * More?

 Maybe if we should have dynamic sizing it could be only for the range. `domain_size` would not be implemented. And the `range_size` would be a function of a vector that the TensorMapping is applied to.

## Reasearch and thinking
 - [ ] Use a trait to indicate if a TensorMapping uses indices with regions.
    The default should be that they do NOT.
        - [ ] What to name this trait? Can we call it IndexStyle but not export it to avoid conflicts with Base.IndexStyle?
 - [ ] Use a trait to indicate that a TensorMapping har the same range and domain?
 - [ ] Rename all the Tensor stuff to just LazyOperator, LazyApplication and so on?
 - [ ] Figure out repeated application of regioned TensorMappings. Maybe an instance of a tensor mapping needs to know the exact size of the range and domain for this to work?
 - [ ] Check how the native julia doc generator works
    - [ ] Check if Vidars design docs fit in there
 - [ ] Create a macro @lazy which replaces a binary op (+,-) by its lazy equivalent? Would be a neat way to indicate which evaluations are lazy without cluttering/confusing with special characters.
 - [ ] Specificera operatorer i TOML eller något liknande?
 H.. H_gamma etc.)
 - [ ] Dispatch in Lower() instead of the type Lower so `::Lower` instead of `::Type{Lower}` ???
 	Seems better unless there is some specific reason to use the type instead of the value.
 - [ ] How do we handle mixes of periodic and non-periodic grids? Seems it should be supported on the grid level and on the 1d operator level. Between there it should be transparent.
 - [ ] Can we have a trait to tell if a TensorMapping is transposable?
 - [ ] Is it ok to have "Constructors" for abstract types which create subtypes? For example a Grids() functions that gives different kind of grids based on input?
