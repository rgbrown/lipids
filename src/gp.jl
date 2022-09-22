using SparseArrays
using LinearAlgebra

# EQP for our particular problem. Equality constraints are A, and
# additionally a sum of x = d constraint. Matrix is passed in as transpose,
# because of its construction

function gradientprojection(xh, k, d, minconcentration=0)
    # TODO: Allow passing of feasible solution and set from previous
    # iteration as initial condition


    # Projection solution onto the feasible region. 
    # Conservation of mass says sum(x) = d. 
    # x is assumed to be [u; v] and the 
    # lipid length is # k indices
    # minconcentration allows a very small minimum concentration, rather
    # than zero. This will stop the issue with log being called on zero
    n = length(xh)
    N = n÷2

    # Define full set of inequality constraints (as tranpose, because
    # A is stored as a compressed-column sparse matrix) A_full x <= b_full
    At_full, b_full = inequalityconstraints(N, k, minconcentration)

    # Find initial feasible solution 
    x0 = fill(d/n, n)

    # Algorithm starts here
    x = copy(x0)
    Ax = At_full'*x # need this later

    n_constraints = size(At_full)[2]

    # Active set (represented as boolean array). Define initial active set
    S = []

    for k in 1:1000
        # Form At and b for binding constraints + equality constraints
        At = @view At_full[:, S]
        b = @view b_full[S]
    
        # Solve EQP 
        xeqp, nu, nu_m = eqp(At, b, d, xh)
    
        # And now figure out what to do with it!
        
        # Check for feasibility of xeqp
        Axeqp = At_full'*xeqp
    
        infeasible = findall(Axeqp .> b_full)
    
        if isempty(infeasible) # soz about double negative
            ineg = findfirst(z->z < 0, nu)
            if isnothing(ineg)
                #optimal
                x[:] = xeqp
                break
            else
                # Remove from S a constraint with negative nu value
                deleteat!(S, ineg)
    
                # Reset x to xeqp
                x[:] = xeqp
                Ax[:] = At_full'*x
            end
        else
            # We need to find t to maximize 
            t, idx = findmin((b_full[infeasible] - Ax[infeasible]) ./ 
                             (Axeqp[infeasible] - Ax[infeasible]))

            append!(S, infeasible[idx])
            x += t*(xeqp - x)
            Ax[:] = At_full'*x
        end
        println(norm(x - xh))
    end

    return x, S

end


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


function inequalityconstraints(N::Int, k, minconcentration)
    # Construct constraint matrix. Saturation on top, and then
    # nonnegativity constraints
    # We construct it as its transpose though, because we are going to be
    # taking column slices of it
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

