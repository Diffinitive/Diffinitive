"""
    undivided_skewed04(g::TensorGrid, p, direction)
    undivided_skewed04(g::EquidistantGrid, p)

Undivided difference operators approximating the `p`th derivative. The
operators do not satisfy any SBP property and are meant to be used for
building artificial dissipation terms.

The operators and how they are used to create accurate artificial dissipation
is described in "K. Mattsson, M. Svärd, and J. Nordström, “Stable and Accurate
Artificial Dissipation,” Journal of Scientific Computing, vol. 21, no. 1, pp.
57–79, Aug. 2004"
"""
function undivided_skewed04 end

function undivided_skewed04(g::TensorGrid, p, direction)
    D,Dᵀ = undivided_skewed04(g.grids[direction], p)
    return (
        LazyTensors.inflate(D, size(g), direction),
        LazyTensors.inflate(Dᵀ, size(g), direction),
    )
end

function undivided_skewed04(g::EquidistantGrid, p)
    T = eltype(g)
    interior_weights = T.(dissipation_interior_weights(p))

    D  = StencilOperatorDistinctClosures(
        g,
        dissipation_interior_stencil(interior_weights),
        dissipation_lower_closure_stencils(interior_weights),
        dissipation_upper_closure_stencils(interior_weights),
    )
    Dᵀ = StencilOperatorDistinctClosures(
        g,
        dissipation_transpose_interior_stencil(interior_weights),
        dissipation_transpose_lower_closure_stencils(interior_weights),
        dissipation_transpose_upper_closure_stencils(interior_weights),
    )

    return D, Dᵀ
end

function dissipation_interior_weights(p)
   if p == 0
       return (1,)
   end

   return (0, dissipation_interior_weights(p-1)...) .- (dissipation_interior_weights(p-1)..., 0)
end

midpoint(weights) = length(weights)÷2 + 1
midpoint_transpose(weights) = length(weights)+1 - midpoint(weights)

function dissipation_interior_stencil(weights)
    return Stencil(weights..., center=midpoint(weights))
end
function dissipation_transpose_interior_stencil(weights)
    if iseven(length(weights))
        weights = map(-, weights)
    end

    return Stencil(weights..., center=midpoint_transpose(weights))
end

dissipation_lower_closure_size(weights) = midpoint(weights) - 1
dissipation_upper_closure_size(weights) = length(weights) - midpoint(weights)

function dissipation_lower_closure_stencils(interior_weights)
    stencil(i) = Stencil(interior_weights..., center=i)
    return ntuple(i->stencil(i), dissipation_lower_closure_size(interior_weights))
end

function dissipation_upper_closure_stencils(interior_weights)
    center(i) = length(interior_weights) - dissipation_upper_closure_size(interior_weights) + i
    stencil(i) = Stencil(interior_weights..., center=center(i))
    return ntuple(i->stencil(i), dissipation_upper_closure_size(interior_weights))
end

function dissipation_transpose_lower_closure_stencils(interior_weights)
    closure = ntuple(i->dissipation_transpose_lower_closure_stencil(interior_weights, i), length(interior_weights))

    N = maximum(s->length(s.weights), closure)
    return right_pad.(closure, N)
end

function dissipation_transpose_upper_closure_stencils(interior_weights)
    closure = reverse(ntuple(i->dissipation_transpose_upper_closure_stencil(interior_weights, i), length(interior_weights)))

    N = maximum(s->length(s.weights), closure)
    return left_pad.(closure, N)
end


function dissipation_transpose_lower_closure_stencil(interior_weights, i)
    w = ntuple(k->interior_weights[i], dissipation_lower_closure_size(interior_weights))

    for k ∈ i:-1:1
        w = (w..., interior_weights[k])
    end

    return Stencil(w..., center = i)
end

function dissipation_transpose_upper_closure_stencil(interior_weights, i)
    j = length(interior_weights)+1-i
    w = ntuple(k->interior_weights[j], dissipation_upper_closure_size(interior_weights))

    for k ∈ j:1:length(interior_weights)
        w = (interior_weights[k], w...)
    end

    return Stencil(w..., center = length(interior_weights)-midpoint(interior_weights)+1)
end
