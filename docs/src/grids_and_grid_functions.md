# Grids and grid functions

The submodule `Grids` aims to provide types and logic for all types of grids that are useful for implementing summation-by-parts difference methods. It provides an abstract top level type `Grid` which defines a broad interface for how a general grid is supposed to work. Currently only equidistant grids are supported, but the basic structure supports implementations of curvilinear grids, multi-block grids, periodic grids and much more.

The module also has functionality for creating and working with grid functions.

## Interface for grids
All grids are expected to work as a grid function for the coordinate function, and thus implements Julia's Indexing- and Iteration-interfaces. Notably they are *not* abstract arrays because that inteface is too restrictive for the types of grids we wish to implement.


## Plotting
Plotting of grids and grid functions is supported through a package extension with Makie.jl.

For grids we have:
* `plot(::Grid{<:Any,2})` (same as `lines`)
* `lines(::Grid{<:Any,2})`
* `scatter(::Grid{<:Any,2})`

For 1D grid functions we have:
* `plot(::Grid{<:Any,1}, ::AbstractVector)` (same as `lines`)
* `lines(::Grid{<:Any,1}, ::AbstractVector)`
* `scatter(::Grid{<:Any,1}, ::AbstractVector)`

For 2D grid functions we have:
* `plot(::Grid{<:Any,2}, ::AbstractArray{<:Any,2})` (constructs a 2d mesh)
* `surface(::Grid{<:Any,2}, ::AbstractArray{<:Any,2})`

## To write about
<!-- # TODO: -->
* Grid functions
  * Basic structure
     * Indexing
  * Curvilinear
  * Multiblock
  * Vector valued grid functions
