# Operator file format

The intention is that Sbplib.jl should be a general and extensible framework
for working with finite difference methods. It therefore includes a set of
tools for storing and sharing operator definitions as well as a set of widely
used operators.

## Using the included operators

Most users will likely access the included operators by simply passing the
filename of the wanted operator set to the appropriate function.  The location
of the included stencil sets can be computed using
[`sbp_operators_path`](@ref).
```@meta
# TODO: provide examples of functions to pass the files to
```
Advanced user might want to get access to the individual objects of an
operator file. This can be accomplished using functions such as
* [`read_stencil_set`](@ref)
* [`parse_scalar`](@ref)
* [`parse_stencil`](@ref)
* [`parse_tuple`](@ref)

When parsing operator objects they are interpreted using `Rational`s and
possibly have to be converted to a desired type before use. This allows
preserving maximum accuracy when needed.
```@meta
# TBD: "possibly have to be converted to a desired type before use" Is this the case? Can it be fixed?
```

## File format
The file format is based on TOML and can be parsed using `TOML.parse`. A file
can optionally start with a `[meta]` section which can specify things like who
the author was, a description and how to cite the operators.

After the `[meta]` section one or more stencil sets follow, each one beginning
with `[[stencil_set]]`. Each stencil set should include descriptors like
`order`, `name` or `number_of_bondary_points` to make them unique within the
TOML-file. What descriptors to use are up to the author of the file to decide.

Beyond identifying information the stencil set can contain any valid TOML.
This data is then parsed by the functions creating specific operators like
``D_1`` or ``D_2``.

### Numbers
Number can be represented as regular TOML numbers e.g. `1`, `-0.4` or
`4.32e-3`. Alternatively they can be represented as strings which allows
specifying fraction e.g. `"1/2"` or `"0"`.

All numbers are accurately converted to `Rational`s when using the
[`parse_scalar`](@ref) function.

### Stencils
Stencils are parsed using [`parse_stencil`](@ref). They can be specified
either as a simple arrays
```toml
stencil = ["-1/2","0", "1/2"]
```
which assumes a centered stencil. Or as a TOML inline table
```toml
stencil =  {s = ["-24/17", "59/34", "-4/17", "-3/34", "0", "0"], c = 1},
```
which allows specifying the center of the stencil using the key `c`.

## Creating your own operator files
Operator files can be created either to add new variants of existing types of
operators like ``D_1`` or ``D_2`` or to describe completely new types of
operators like for example a novel kind of interpolation operator. In the
second case new parsing functions are also necessary.

The files can then be used to easily test or share different variants of
operators.
