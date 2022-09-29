include("gp.jl")
using Printf
import .Gp

function foo()
    L = 30
    dx = 0.1
    lipidlength = 5
    k = Int(round(lipidlength/dx))
    c0 = 0.0625
    m = 2
    minconc = 1e-5
    
    
    A, b, C, d, f, df, y0, B, plotuv = Gp.initialise(L, dx, lipidlength, c0, m, minconcentration=minconc)
    
    n, nconstraints = size(A)
    n = length(y0)
    N = setdiff(1:nconstraints, B)
    
    y = copy(y0)
    plotuv(y, true)
    debug = true
    for j = 1:4000
        debug && @printf("iteration %d: f(y) = %f\n", j, f(y)) 
        y = Gp.descentstep!(y, B, N, A, b, C, d, f, df, debug=true,
                            tol=1e-11, αmax=0.5, maxits=100)
    end
    return A, b, C, d, f, df, y0, y, B, N, plotuv
end
A, b, C, d, f, df, y0, y, B, N, plotuv = foo()



