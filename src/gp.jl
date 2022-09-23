using SparseArrays
using LinearAlgebra
import DSP
using Plots

""" 
    lipidbilayer(L, dx, lipidlength, c0, m; minconcentration=0.0, alpha=1, sigma=5)

Solve for lipid bilayer by gradient projection.

`L`:                Periodic domain ``[-L, L]``
`dx`:               Domain spacing. Make sure dx divides L
`lipidlength`:      Length of lipid molecule (multiple of dx)
`c0`:               Background concentration
`m`:                Total lipid mass
`minconcentration`: Minimum concentration of lipid. A small value instead
                    of 0 prevents log(0) ever being called
"""
function lipidbilayer(L, dx, lipidlength, c0, m; minconcentration=0.0, alpha=1, sigma=5, uc=0, vc=0)
    x = -L:dx:L
    N = length(x)
    n = 2*x

    k = Int(round(lipidlength/dx))
    # Right hand side for sum(u + v) = d
    d = m/dx + 2*N*c0

    # Define inequality constraint matrix and RHS
    At_full, b_full = inequalityconstraints(N, k, minconcentration)

    # Construct kernel
    xker = -10:dx:10
    K = kappa(xker)
    alpha = 1

    function df(y)
        dy = zeros(size(y))
        u = @view y[1:N]
        v = @view y[N+1:end]
        kuv = convwrap(u + v, K)
        dy[1:N] = log.(u) - 2*alpha*kuv
        dy[N+1:end] = log.(v) - 2*alpha*kuv
        return dy
    end

    function f(y)
        u = @view y[1:N]
        v = @view y[N+1:end]
        F = sum(u.*log.(u) + v.*log.(v) + 
                alpha*(1 .- (u + v)).*convwrap(u + v, K))*dx
    end

    function proj(y, y0=[], S=Int[])
        y, S = projectfeasible(y, At_full, b_full, d, x0=y0, S=S)
        return y, S
    end

    # Initial conditions
    u0 = 1/(sigma*sqrt(2*pi))*exp.(-0.5*(x .- uc).^2/sigma^2) 
    v0 = 1/(sigma*sqrt(2*pi))*exp.(-0.5*(x .- vc).^2/sigma^2) 
    u0 = m/(2*sum(u0)*dx)*u0 .+ c0
    v0 = m/(2*sum(v0)*dx)*v0 .+ c0
    y0 = [u0; v0]

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

    projectedgradient(f, df, proj, y0)
end


function projectedgradient(fun, gradfun, projfun, x, alpha=0.5, beta=0.8, S=[])
    # Assuming x is feasible

    # Choose a descent direction
    t = 1
    for k = 1:100
      f = fun(x)
      println(f)
      dx = -gradfun(x)
      # Normalise for conservation of mass
      dx -= sum(dx)/length(dx)
      foo = -dot(dx, dx)

      # Line search
      t = 1.0
      for j = 1:100
          xnew, S = projfun(x + t*dx)
          fnew = fun(xnew)
          if fnew < f + alpha*t*foo:
              x = x + t*dx
              break
          end
          t *= beta
      end
  end
end


function findbilayer()
    # Set up our objective function 

    gradfun, fun, At_full, b_full, d, y0, plotuv = lipidbilayer(30, 0.1, 5, 0.0625, 2, minconcentration=1e-5, sigma=0.5)


    df = zeros(size(y0))

    y, S = projectfeasible(y0, At_full, b_full, d)
end



function kappa(s)
    0.5*exp.(-abs.(s))
end


"""
    convwrap(x, w)

    `x` is an array on a periodic domain
    `w` is the window function to convolve with
"""
function convwrap(x, w)
    n = size(x)[1]
    f = size(w)[1]
    
    # f should be odd
    # Add on (f - 1)/2 entries of x to each end
    xnew = vcat(x[(n - (f-1)÷2 + 1):end], x, x[1:(f-1)÷2])

    # Perform the convolution and truncate the tails
    y = DSP.conv(xnew, w)[f:end-f+1]
end



"""
    projectfeasible(xh, At_full, b_full, d, x0=[], S=Int[], maxits=1000)

Project a vector `xh` onto the feasible region. 

`x0`, `S` : Initial feasible solution and binding constraints.
If `x0` is not provided, any provided `S` is ignored.

Returns `x`, `S`, the solution along with the active set of binding constraints

"""
function projectfeasible(xh, At_full, b_full, d; x0=[], S=Int[], maxits=1000)
    debug = false
    # Create feasible initial condition, if required. Requires d to be
    # small enough
    n = length(xh)
    if isempty(x0)
        x0 = fill(d/n, n)
        S = Int[]
    end

    Ax = At_full'*x0

    x = copy(x0)
    for k in 1:maxits
        # Form At and b for binding constraints + equality constraints
        At = @view At_full[:, S]
        b = @view b_full[S]
    
        # Solve EQP - solution maybe inside or outside feasible region
        xeqp, nu, nu_m = eqp(At, b, d, xh)
        
        # Check for feasibility of xeqp
        Axeqp = At_full'*xeqp
    
        # setdiff required becuase existing ones might not hold exactly to
        # machine precision
        infeasible = setdiff(findall(Axeqp .> b_full), S)
    
        if isempty(infeasible) # i.e. if solution is feasible
            debug && println("feasible") 
            ineg = findfirst(z->z < 0, nu)
            if isnothing(ineg) # no negative Lagrange multipliers -> optimal
                x[:] = xeqp
                break
            else 
                # Remove from S a constraint with negative nu value
                debug && println("removing constraint") 
                deleteat!(S, ineg)
    
                # Reset x to xeqp
                x[:] = xeqp
                Ax[:] = At_full'*x
            end
        else
            debug && println("infeasible")
            # Move from x toward x_eqp until we hit first constraint
            # x + t(x_eqp - x)
            # Find t to make each infeasible inequality at x_eqp equality,
            # and choose smallest one.
            t, idx = findmin((b_full[infeasible] - Ax[infeasible]) ./ 
                             (Axeqp[infeasible] - Ax[infeasible]))

            # Add corresponding constraint, update x and Ax
            debug && println("Adding constraint ", infeasible[idx])
            append!(S, infeasible[idx])
            x += t*(xeqp - x)
            Ax[:] = At_full'*x
        end
    end
    return x, S
end






"""
    eqp(At, b, d, xh)

Solve the EQP for the lipids problem, minimising distance from x to xh
subject to equality constraints.

Constraints are `At'*x = b` and `sum(x) = d`. 

Returns 
`x`: solution
`nu`: Lagrange multipliers for constraints in At
`nu_m`: Lagrange multiplier for conservation of mass constraint
"""
function eqp(At, b, d, xh)
    n, m = size(At)

    # Form KKT matrix
    K = [sparse(I, n, n) At ones(n, 1); 
         At' spzeros(m, m+1); 
         ones(1, n) spzeros(1, m+1)]
    rhs = vcat(xh, b, d)

    # Solve prolem
    y = K \ rhs
    
    x, nu, nu_m = y[1:n], y[n+1:end-1], y[end]
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


