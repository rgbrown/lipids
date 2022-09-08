# Trying a Julia implementation to see if we can get more speed

# First up - need to write the wrapped convolution method
import DSP
using DifferentialEquations

function conv_wrap(x, w)
    n = size(x)[1]
    f = size(w)[1]
    
    # f should be odd
    # Add on (f - 1)/2 entries of x to each end
    xnew = vcat(x[(n - (f-1)÷2 + 1):end], x, x[1:(f-1)÷2])

    # Perform the convolution and truncate the tails
    y = DSP.conv(xnew, w)[f:end-f+1]
end

# Problem parameters
L = 30
dx = 0.1
x = -L:dx:L
N = size(x)[1]

eps = 5
K = 1e-5
alpha = 8
c0 = 0.0625

xker = -10:dx:10
kappa = 0.5*exp.(-abs.(xker))

xu = -5
xv = 5
sigma = 20
m = 1

neps = Int(round(eps/dx))
function tauplus(y)
    circshift(y, neps)
end

function tauminus(y)
    circshift(y, -neps)
end

function convolvewithkappa(y)
    dx*conv_wrap(y, kappa)
end

function rhs(y, p, t)
    u = y[1:N]
    v = y[N+1:end]

    mu = u .+ tauminus(u) .+ v .+ tauplus(v) .- 1
    kuv = convolvewithkappa(u .+ v)

    uti = -log.(u) .+ 2*alpha*kuv .- K ./ (mu.^2) .- K ./ (tauplus(mu).^2)
    vti = -log.(v) .+ 2*alpha*kuv .- K ./ (mu.^2) .- K ./ (tauminus(mu).^2)

    lam = -(sum(uti) + sum(vti))/(2*N)

    return vcat(uti .+ lam, vti .+ lam)
end

u0 = c0 .+ m/(2*sigma*sqrt(pi)) * exp.(-0.5*((x .- xu)/sigma).^2)
v0 = c0 .+ m/(2*sigma*sqrt(pi)) * exp.(-0.5*((x .- xv)/sigma).^2)
y0 = vcat(u0, v0)

tspan = (0.0, 2.0)

prob = ODEProblem(rhs, y0, tspan)
solve(prob)

