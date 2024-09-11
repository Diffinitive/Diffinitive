module DiffinitivePlotsExt

using Diffinitive.Grids
using Plots

@recipe f(::Type{<:Grid}, g::Grid) = map(Tuple,g)[:]

@recipe function f(c::Chart{2,<:Rectangle}, n=5, m=n; draw_border=true, bordercolor=1)
    Ξ = parameterspace(c)
    ξs = range(limits(Ξ,1)..., n)
    ηs = range(limits(Ξ,2)..., m)

    label := false
    seriescolor --> 2
    for ξ ∈ ξs
        @series adapted_curve_grid(η->c((ξ,η)),limits(Ξ,1))
    end

    for η ∈ ηs
        @series adapted_curve_grid(ξ->c((ξ,η)),limits(Ξ,2))
    end

    if ~draw_border
        return
    end

    for ξ ∈ limits(Ξ,1)
        @series begin
            linewidth --> 3
            seriescolor := bordercolor
            adapted_curve_grid(η->c((ξ,η)),limits(Ξ,1))
        end
    end

    for η ∈ limits(Ξ,2)
        @series begin
            linewidth --> 3
            seriescolor := bordercolor
            adapted_curve_grid(ξ->c((ξ,η)),limits(Ξ,2))
        end
    end
end

function adapted_curve_grid(g, minmax)
    t1, _ = PlotUtils.adapted_grid(t->g(t)[1], minmax)
    t2, _ = PlotUtils.adapted_grid(t->g(t)[2], minmax)

    ts = sort(vcat(t1,t2))

    x = map(ts) do t
        g(t)[1]
    end
    y = map(ts) do t
        g(t)[2]
    end

    return x, y
end

# get_axis_limits(plt, :x)


# ReicpesPipline/src/user_recipe.jl
# @recipe function f(f::FuncOrFuncs{F}) where {F<:Function}

# @recipe function f(f::Function, xmin::Number, xmax::Number)

# _scaled_adapted_grid(f, xscale, yscale, xmin, xmax)

end


