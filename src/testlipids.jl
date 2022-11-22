include("lipids1d.jl")
import .Lipids

# Run all three models on a test set of parameters
Lipids.testlipids()

# L = 10
# c0 = 0.024
# m = 0.05*2*L 
# dx = 0.1 
# lipidlength = 2 
# cmin = 1e-5
# sigma=5
# bilayermodel="bw"
# gamma = 1
# alpha = 3*2/(1 + gamma)
# x, p, u, v, u0, v0 = Lipids.lipidbilayer(L, dx, lipidlength, c0, m, gamma=gamma, alpha=alpha, sigma=sigma, bilayermodel=bilayermodel)
# 
# # Problem is invariant to cyclic shifts, and JuMP tends to produce a solution
# # that's centered at the first index, so do a shift of half the domain length to
# # put it back
# s = length(u) ÷ 2
# display(Lipids.plotuv(x, u0, v0, p, title="initial conditions"))
# display(Lipids.plotuv(x, u, v, p, title="m = $(m)"))


