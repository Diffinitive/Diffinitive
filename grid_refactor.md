# Grids refactor

## Change summary
* `EquidistantGrid` is now only a 1D thing.
* Higher dimensions are supported through `TensorGrid`.
* The old behavior of `EquidistantGrid` has been moved to the function `equidistant_grid`.
* Grids embedded in higher dimensions are now supported through tensor products with `ZeroDimGrid`s.
* Vector valued grid functions are now supported and the default element type is `SVector`.
* Grids are now expected to support Julia's indexing and iteration interface.
* `eval_on` can be called with both `f(x,y,...)` and `f(x̄)`.

## TODO
* Delete this document, move remaining notes to Notes.md
