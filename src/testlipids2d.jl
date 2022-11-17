include("lipids2d.jl")
import .Lipids2D

L = 1
dx = 0.5
lipidlength = 1
c0 = 0.025
m = 10

Lipids2D.lipidbilayer(L, dx, lipidlength, c0, m; 
    cmin=1e-5, alpha=1, sigma = 5, gamma=0.7)




