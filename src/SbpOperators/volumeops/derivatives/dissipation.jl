function undevided_dissipation(g::EquidistantGrid, p, direction)
    T = eltype(g)
    interior_weights = T.(dissipation_interior_weights(p))

    D  = stencil_operator_distinct_closures(
        g,
        dissipation_interior_stencil(interior_weights),
        dissipation_lower_closure_stencils(interior_weights),
        dissipation_upper_closure_stencils(interior_weights),
        direction,
    )
    Dᵀ = stencil_operator_distinct_closures(
        g,
        dissipation_transpose_interior_stencil(interior_weights),
        dissipation_transpose_lower_closure_stencils(interior_weights),
        dissipation_transpose_upper_closure_stencils(interior_weights),
        direction,
    )

    return D, Dᵀ
end

undevided_dissipation(g::EquidistantGrid{1}, p) = undevided_dissipation(g, p, 1)

function dissipation_interior_weights(p)
   if p == 0
       return (1,)
   end

   return (0, dissipation_interior_weights(p-1)...) .- (dissipation_interior_weights(p-1)..., 0)
end

midpoint(weights) = length(weights)÷2 + 1
midpoint_transpose(weights) = length(weights)+1 - midpoint(weights)

dissipation_interior_stencil(weights) =           Stencil(weights..., center=midpoint(weights))
dissipation_transpose_interior_stencil(weights) = Stencil(weights..., center=midpoint_transpose(weights))

dissipation_lower_closure_size(weights) = midpoint(weights) - 1
dissipation_upper_closure_size(weights) = length(weights) - midpoint(weights)

dissipation_lower_closure_stencils(interior_weights) = ntuple(i->Stencil(interior_weights..., center=i                       ), dissipation_lower_closure_size(interior_weights))
dissipation_upper_closure_stencils(interior_weights) = ntuple(i->Stencil(interior_weights..., center=length(interior_weights)-dissipation_upper_closure_size(interior_weights)+i), dissipation_upper_closure_size(interior_weights))

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
