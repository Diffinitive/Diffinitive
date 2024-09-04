module SbplibSparseArrayKitExt

using Sbplib
using Sbplib.LazyTensors

using SparseArrayKit
using Tokens

"""
    SparseArray(t::LazyTensor)

The sparse tensor representation of `t` with range dimensions to the left and
domain dimensions to the right. If `L` is a `LazyTensor`  with range and
domain dimension 2 and `v` a 2-tensor, then `A = SparseArray(t)` is
constructed so that `∑ₖ∑ₗA[i,j,k,l]*v[k,l] == L*v`§
"""
function SparseArrayKit.SparseArray(t::LazyTensor)
    v = ArrayToken(:v, domain_size(t)...)
    return Tokens._to_tensor(t*v, range_size(t), domain_size(t))
end

end
