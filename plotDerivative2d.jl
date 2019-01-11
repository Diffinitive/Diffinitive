g = sbp.grid.EquidistantGrid((100,75),((0, 0), (2pi, 3/2*pi)))
op = sbp.readOperator("d2_4th.txt","h_4th.txt")
Laplace = sbp.Laplace2D(g,1,op)

init(x,y) = sin(x) + cos(y)
v = sbp.grid.evalOn(g,init)

u = zeros(length(v))

sbp.apply!(Laplace,u,v)

@show u
@show u'*u

sbp.grid.plotgridfunction(g,u)

