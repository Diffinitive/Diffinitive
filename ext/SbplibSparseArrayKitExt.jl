module SbplibSparseArrayKitExt

using Sbplib
using Sbplib.LazyTensors

using SparseArrayKit
using Tokens

function SparseArrayKit.SparseArray(t::LazyTensor)
    v = ArrayToken(:v, domain_size(t)...)
    return Tokens._to_tensor(t*v, range_size(t), domain_size(t))
end

end
