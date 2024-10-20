# Matrix and tensor representations

Sparse matrix and sparse tensor representations of lazy tensors can be constructed by loading [Tokens.jl](http://) and one of SparseArrays.jl or [SparseArrayKit.jl](http://). Through package extensions the following methods `sparse(::LazyTensor)` and `SparseArray(::LazyTensor)` are provided.

<!-- TODO figure out how to add the docstrings here --/>
