g = sbp.Grid.EquidistantGrid((200,), (0.0,), (2pi,))
op =sbp.readOperator("d2_4th.txt","h_4th.txt")
Laplace = sbp.Laplace(g,1.0,op)

init(x) = cos(x)
v = sbp.Grid.evalOn(g,init)
u = zeros(length(v))

sbp.apply!(Laplace,u,v)

@show u
sbp.Grid.plotgridfunction(g,u)
