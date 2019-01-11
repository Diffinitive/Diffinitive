g = sbp.grid.EquidistantGrid(200,(0, 2pi))
op =sbp.readOperator("d2_4th.txt","h_4th.txt")
Laplace = sbp.Laplace1D(g,1,op)

init(x) = sin(x)
v = sbp.Grid.evalOn(g,init)
u = zeros(length(v))

sbp.apply!(Laplace,u,v)

@show u
sbp.grid.plotgridfunction(g,u)
