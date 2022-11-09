module Lipids
export bp

import FFTW
using SparseArrays
using LinearAlgebra

function bp(L, dx, lipidlength, c0, m; 
    minconcentration=0.0, alpha=1, sigma=5, uc=0, vc=0)

    # Construct the domain
    x = -L:dx:L
    N = length(x)
    n = 2*N

    k = Int(round(lipidlength/dx))

    # Equality constraint matrices for C'x = d
    C = ones(n, 1)
    d = m/dx + 2*N*c0

    # Inequality constraint matrix and RHS
    A, b = inequalityconstraints(N, k, minconcentration)

    # Construct kernel
    xker = -10:dx:10
    K = kappa(xker)
    sumK = sum(K)

    # Set up convolution function with kernel
    convolvewithkappa = makeconv(K, N)

    # Set up right hand side and gradient
    function fbp(y)
        u = @view y[1:N]
        v = @view y[N+1:end]

        F = sum(u.*log.(u) + v.*log.(v) + 
                alpha*(1 .- (u + v)).*convolvewithkappa(u + v))*dx
    end

    function dfbp(y)
        dy = zeros(size(y))
        u = @view y[1:N]
        v = @view y[N+1:end]
        kuv = convolvewithkappa(u + v)
        dy[1:N] = (1 + alpha*sumK) .+ log.(u) - 2*alpha*kuv
        dy[N+1:end] = (1 + alpha*sumK) .+ log.(v) - 2*alpha*kuv
        return dy*dx
    end

    function fwht(y)
        u = @view y[1:N]
        v = @view y[N+1:end]

        F = sum(u.*log(u) + v.*log(v) + 
            alpha*(1 - u - v - tauminus(u) - tauplus(v)) .* 
            convolvewithkappa(1 .- u - v))*dx
    end

    # TODO do this
    function dfwht(y)
        dy = zeros(size(y))
        u = @view y[1:N]
        v = @view y[N+1:end]
    end

    function plotuv(y, keep=false)
        u = @view y[1:N]
        v = @view y[N+1:end]
        tails = u + v
        heads = circshift(u, -k) + circshift(v, k)
        if keep
            plot!(x, tails)
            plot!(x, heads)
        else
            plot(x, tails, label="tails")
            plot!(x, heads, label="heads")
        end
    end

    # Initial conditions
    u0 = 1/(sigma*sqrt(2*pi))*exp.(-0.5*(x .- uc).^2/sigma^2) 
    v0 = 1/(sigma*sqrt(2*pi))*exp.(-0.5*(x .- vc).^2/sigma^2) 
    u0 = m/(2*sum(u0)*dx)*u0 .+ c0
    v0 = m/(2*sum(v0)*dx)*v0 .+ c0
    y0 = [u0; v0]

    return A, b, C, d, fbp, dfbp, y0, plotuv
end

"""
    inequalityconstraints(N, k, minconcentration)

Construct inequality constraint matrix for lipids problem. 
`N` is the number of points in the domain
`k` is the lipid length in grid points
`minconcentration` allows the minimum concentration to be specified. A very small number can be provided to approximate zero

Returns
`At_full`,
`b_full`
Such that all inequality constraints are `At_full' * x .<= b_full` 
Note that the transpose is returned, as sparse matrices are stored in
compressed column format in Julia
"""
function inequalityconstraints(N::Int, k, minconcentration)
    # Construct constraint matrix. Saturation on top, and then
    # nonnegativity constraints
    # We construct the transpose as Julia uses compressed column sparse
    # matrices
    n = 2*N
    w = 1:N
    rows = [w mod.(w .+ (k-1), N) .+ 1 w .+ N mod.(w .- (k+1), N) .+ (N + 1)]
    cols = [w w w w]
    vals = ones(N, 4)

    At_full = [sparse(rows[:], cols[:], vals[:]) -sparse(I, n, n)]

    # Construct right hand sides 
    b_full = [ones(N); fill(-minconcentration, n)]
    return At_full, b_full
end

# Kernel function
function kappa(s)
    0.5*exp.(-abs.(s))
end

"""
    makeconv(w, n)

    Make a function that convolves the kernel w with a vector of length n on a periodic domain
    
    Returns:
        a function f(x) which performs the convolution
"""
function makeconv(w, n)
    nw = length(w)
    if nw > n
        @error "kernel length cannot be larger than n"
    else
        # circshift to make the centre of the kernel at index 1
        # NB: another option would be to define our kernel function so that this
        # happens automatically. Probably should be done, but leaving for now
        w = circshift(cat(w, zeros(n - nw), dims=1), -(nw ÷ 2))
    end

    F = FFTW.plan_rfft(rand(n))
    W = F * w
    R = FFTW.plan_irfft(W, n)

    function convwrap(x)
        return R*(W.*(F*x))
    end
    return convwrap
end
end