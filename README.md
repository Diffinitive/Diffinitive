# Sbplib

## Running tests
To run all tests simply run
```
(@v1.5) pkg> activate .
(Sbplib) pkg> test
```

If you want to run tests from a specific file in `test/`, you can do
```
julia> using Pkg
julia> Pkg.test(test_args=["[glob pattern]"])
```
For example
```
julia> Pkg.test(test_args=["SbpOperators/*"])
```
to run all test in the `SbpOperators` folder, or
```
julia> Pkg.test(test_args=["*/readoperators.jl"])
```
to run only the tests in files named `readoperators.jl`.
Multiple filters are allowed and will cause files matching any of the provided
filters to be run. For example
```
Pkg.test(test_args=["*/lazy_tensor_operations_test.jl", "Grids/*"])
```
will run any file named `lazy_tensor_operations_test.jl` and all the files in the `Grids` folder.
