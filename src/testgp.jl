include("gp.jl")
import .Gp

L = 30
dx = 0.1
lipidlength = 5
k = Int(round(lipidlength/dx))
c0 = 0.0625
m = 2
minconc = 1e-5

A, b, C, d, f, df, y0, B, x = Gp.initialise(L, dx, lipidlength, c0, m, minconcentration=minconc)

n = length(y0)
N = setdiff(1:n, B)

g = -df(y0)
dir = Gp.projectgradient(g, [A[:, B] C], [b[B]; d])
t, index = Gp.findconstraint(y0, dir, A[:,N], b[N])
alpha = Gp.linesearch(y0, dir, f(y0), df(y0), f, t, maxits=10, c=0.2)
if t == alpha
    push!(B, N[index])
    deleteat!(N, index)
end

Gp.plotuv(x, y1, k)
