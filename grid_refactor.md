# Grids refactor

## Goals before merging
* A somewhat clear path towards multi-block grids and their grid functions.
* A somewhat clear path towards implementations of div() and rot() or the elastic operator (See Notes.md)

## Change summary
* `EquidistantGrid` is now only a 1D thing.
* Higher dimensions are supported through `TensorGrid`.
* The old behavior of `EquidistantGrid` has been moved to the function `equidistant_grid`.
* Grids embedded in higher dimensions are now supported through tensor products with `ZeroDimGrid`s.
* Vector valued grid functions are now supported and the default element type is `SVector`.
* Grids are now expected to support Julia's indexing and iteration interface.
* `eval_on` can be called with both `f(x,y,...)` and `f(x̄)`.


## TODO
* Add benchmarks or allocation tests for eval_on and indexing grids.
* Add benchmarks for range type in EquidistantGrid. (LinRange vs StepRange)
* Write about the design choices in the docs.
* Merge and run benchmarks

* Clean out Notes.md of any solved issues
* Delete this document, move remaining notes to Notes.md

## Frågor

### Implement the tensor product operator for grids?
Yes!
This could be a useful way to create grids with mixes of different kinds of 1d grids. An example could be a grid which is periodic in one direction and bounded in one.
