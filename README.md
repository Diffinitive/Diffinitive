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
Benchmarks are defined in `benchmark/` and use the tools for benchmark suites in BenchmarkTools.jl. The format is compatible with PkgBenchmark.jl which helps with running the suite, comparing results and presenting the results in a readable way. There are custom functions included for running the benchmarks in this Mercurial repository.

`benchmark/` contains a julia environment with the necessary packages for working with the benchmarks.

To run the benchmarks, either use `make` run them manually from the REPL, as explained further below.

Using `make` there are four targets for benchmarks
```shell
make benchmark                                      # Runs the suite for the current working directory
make benchmarkrev REV=rev                           # Runs the suite at the specified revision
make benchmarkcmp TARGET=target BASELINE=baseline   # Compares two revisions
make cleanbenchmark                                 # Cleans up benchmark tunings and results
```
Here `rev`, `target` and `baseline` are any valid Mercurial revision specifiers. Note that `make benchmarkrev` and `make benchmarkcmp` will fail if you have pending changes in your repository.


Alternatively, the benchmarks can be run from the REPL. To do this, first activate the environment in `benchmark/` then include the file `benchmark_utils.jl`. The suite can then be run using the function `main` in one of the following ways

```julia
main()                  # Runs the suite for the current working directory
main(rev)               # Runs the suite at the specified revision
main(target, baseline)  # Compares two revisions
```

Again, `rev`, `target` and `baseline` are any valid Mercurial revision specifiers. Note that `main(rev)` and `main(target, baseline)` will fail if you have pending changes in your repository.

PkgBenchmark can also be used directly.

```julia
using PkgBenchmark
import Sbplib
r = benchmarkpkg(Sbplib)

export_markdown(stdout, r)
```

## Generating and using the documentation
Generating the documentation can be done using either `make` or through activating the `docs` environment and including the script `docs/make.jl` at the REPL.

Using `make` there are four targets for documentation
```shell
make docs          # generates files suitable for webserver deployment, i.e with `prettyurls=true`
make localdocs     # generates files suitable for local viewing in a web browser, i.e `prettyurls=false`
make opendocs      # build and view documentation locally
make cleandocs     # cleans up generated files
```

Alternatively, to view the documentation locally simply open `docs/build/index.html` in your web browser. When including the `docs/make.jl` script `prettyurls` is set to `false` by default.

Including `docs/make.jl` from the REPL may be preferable when repeatedly building the documentation since this avoids compilation latency.
