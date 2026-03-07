function divergence(g::TensorGrid, stencil_set)
    Ds = map(1:ndims(g)) do i
        first_derivative(g, stencil_set, i)
    end

    return VectorDotTensor(Ds)
end

function gradient(g::TensorGrid, stencil_set)
    Ds = map(1:ndims(g)) do i
        first_derivative(g, stencil_set, i)
    end

    return VectorTensor(Ds)
end
