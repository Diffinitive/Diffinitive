# Notes

## Known size of range and domain?
It might be a good idea let tensormappings know the size of their range and domain as a constant. This probably can't be enforced on the abstract type but maybe we should write our difference operators this way. Having this as default should clean up the thinking around adjoints of boundary operators. It could also simplify getting high performance out of repeated application of regioned TensorMappings.
Is there any reason to use a trait to differentiate between fixed size and unknown size?

## Test setup
Once we figure out how to organize the subpackages we should update test folders to Project. As of writing this there seems to be and issue with this approach combined with dev'ed packages so we can't do it yet. It seems that Pkg might fix this in the future.

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
 H.. H_gamma etc.)
