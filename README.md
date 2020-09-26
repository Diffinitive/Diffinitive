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
julia> Pkg.test(test_args=["testLazyTensors"])
```
This works by using the `@includetests` macro from the [TestSetExtensions](https://github.com/ssfrr/TestSetExtensions.jl) package. For more information, see their documentation.
