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

## Running benchmarks
Benchmarks are defined in `benchmark/` use the tools for benchmark suites in BenchmarkTools.jl
The format is compatible with PkgBenchmark.jl which helps with running the suite, comparing results and presenting the results in a readable way.

`benchmark/` contains a julia environment with the necessary packages for working with the benchmarks.

`activate benchmark`

```julia
using PkgBenchmark
import Sbplib
r = benchmarkpkg(Sbplib)

export_markdown(stdout, r)
```

#TODO: Clean this up


## Generating and using the documentation
Generating the documentation can be done using either `make` or through activating the `docs` environment and including the script `docs/make.jl` at the REPL.

Using `make` there are three targets
```shell
make docs
make localdocs
make opendocs
make help
```
The first variant generates files suitable for webserver deployment, i.e with `prettyurls=true`. The second generates files sutible for local viewing in a web browser, i.e `prettyurls=false`. To view the documentation locally simply open `docs/build/index.html` in your web browser. The documentation can be automatically built and opened using
```shell
make opendocs
```

When including the `docs/make.jl` script `prettyurls` is set to `false` by default.

Including `docs/make.jl` from the REPL may be preferable when repeatadely building the documentation since this avoids compilation latency.
