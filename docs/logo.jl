using Plots
using Colors

logocolors = Colors.JULIA_LOGO_COLORS

gr()
# pgfplotsx()

polar(r,θ) = (r*cos(θ), r*sin(θ))

plt = plot(;
    xlim=(-1,1),
    ylim=(-1,1),
    aspect_ratio=1.,
    legend=false,
    axis=([],false),
    grid=false,
    dpi=600,
    size=(250,250),
    background_color = :transparent,
)

origo = (0,0);
r = 0.6

markersize=24
markerstrokewidth=3
markerstrokecolor=colorant"#333"

for θ ∈ π/2 .+ 2π/3*(0:2)
    plot!([origo, polar(r,θ)];
        color = colorant"#ccc",
        linewidth=4,
    )
end

scatter!(origo;
    color = logocolors.blue,
    markersize,
    markerstrokewidth,
    markerstrokecolor,
)


colors = [logocolors.green, logocolors.red, logocolors.purple]

for i ∈ 0:2
    θ = π/2 + 2π/3*i
    scatter!(polar(r,θ);
        color = colors[i+1],
        markersize,
        markerstrokewidth,
        markerstrokecolor,
    )
end







plt2 = plot(;
    xlim=(-1,1),
    ylim=(-1,1),
    aspect_ratio=1.,
    legend=false,
    axis=([],false),
    grid=false,
    dpi=600,
    size=(300,300),
    background_color = :transparent
)

origo = (0,0);

r = 0.6
markersize=24
markerstrokewidth=3
markerstrokecolor=colorant"#333"

for θ ∈ 2π/4*(0:3)
    plot!([origo, polar(r,θ)];
        color = colorant"#CCC",
        linewidth=4,
    )
end

scatter!(origo;
    color = colorant"#FFF",
    markersize,
    markerstrokewidth,
    markerstrokecolor,
)


colors = [logocolors.blue, logocolors.red, logocolors.green, logocolors.purple]

for i ∈ 0:3
    θ = 2π/4*i
    scatter!(polar(r,θ);
        color = colors[i+1],
        markersize,
        markerstrokewidth,
        markerstrokecolor,
    )
end

savefig(plt, "tri.png")
savefig(plt2, "quad.png")
savefig(plt, "logo.svg")
