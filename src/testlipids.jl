module Tst
include("lipids1d.jl")
using .Lipids

L = 30
c0 = 0.0625
m = 2
dx = 0.1
lipidlength = 5
minconc = 1e-5


A, b, C, d, f, df, y0, plotuv = bp(L, dx, lipidlength, c0, m, minconcentration = minconc)

end
