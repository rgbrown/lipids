using SparseArrays
using LinearAlgebra

# EQP for our particular problem. Equality constraints are A, and
# additionally a sum of x = d constraint. Matrix is passed in as transpose,
# because of its construction
function eqp(At, b, d, xh)
    n, m = size(At)

    # Form KKT matrix
    K = KKTmatrix(At)

    rhs = vcat(xh, b, d)

    # Solve prolem
    y = K \ rhs
    x, nu, nu_m = y[1:n], y[n+1:end-1], y[end]
end

function gradientprojection(xh, k, d, minconcentration=0)
    # Projection solution onto the feasible region. 
    # Conservation of mass says sum(x) = d. 
    # x is assumed to be [u; v] and the 
    # lipid length is # k indices
    n = length(xh)
    N = n÷2

    At_full, b_full = inequalityconstraints(N, k, minconcentration)

    # Find initial feasible solution 
    x0 = fill(d/n, n)

    # Initial binding constraint set
    S = []

    # Form At and b
    At = @view At_full[:, S]
    b = @view b_full[S]
    K = KKTmatrix(At)

    x, nu, nu_m = eqp(At, b, d, x0)

    return At, b, K, x, nu, nu_m

end

function KKTmatrix(At)
    n, m = size(At)
    K = [sparse(I, n, n) At ones(n, 1); 
         At' spzeros(m, m+1); 
         ones(1, n) spzeros(1, m+1)]
    return K
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
    b_full = [ones(N, 1); fill(-minconcentration, n)]
    return At_full, b_full
end

