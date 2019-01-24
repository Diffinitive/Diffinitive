g = sbp.Grid.EquidistantGrid((100,75), (0.0, 0.0), (2pi, 3/2*pi))
op = sbp.readOperator("d2_4th.txt","h_4th.txt")
Laplace = sbp.Laplace(g, 1.0, op)

init(x,y) = sin(x) + cos(y)
v = sbp.Grid.evalOn(g,init)

u = zeros(length(v))

sbp.apply!(Laplace,u,v)

@show u
@show u'*u

sbp.Grid.plotgridfunction(g,u)

