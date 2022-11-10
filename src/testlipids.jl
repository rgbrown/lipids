include("lipids1d.jl")
using .Lipids

L = 30
c0 = 0.0
m = 30 
dx = 0.2
lipidlength = 5
cmin = 1e-5
sigma=5

x, p, u, v, u0, v0 = lipidbilayer(L, dx, lipidlength, c0, m, sigma=sigma)
# Because I'm not specifying an initial condition, it tends to produce a bilayer
# centred on the end of the domain (problem is invariant to cyclic shifts). so
# shift it back to the middle
s = length(u) ÷ 2
display(plotuv(x, u0, v0, p, title="initial conditions"))
display(plotuv(x, circshift(u, s), circshift(v,s), p, title="m = $(m)"))


