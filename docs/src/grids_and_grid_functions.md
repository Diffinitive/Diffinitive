# Grids and grid functions

The submodule `Grids` aims to provide types and logic for all types of grids that are useful for implementing summation-by-parts difference methods. It provides an abstract top level type `Grid` which defines a broad interface for how a general grid is supposed to work. Currently only equidistant grids are supported, but the basic structure supports implementations of curvilinear grids, multi-block grids, periodic grids and much more.

The module also has functionality for creating and working with grid functions.

## Interface for grids
All grids are expected to work as a grid function for the coordinate function, and thus implements Julia's Indexing- and Iteration-interfaces. Notably they are *not* abstract arrays because that inteface is too restrictive for the types of grids we wish to implement.

## To write about
<!-- # TODO: -->
* Grid functions
  * Basic structure
     * Indexing
  * Curvilinear
  * Multiblock
  * Vector valued grid functions
