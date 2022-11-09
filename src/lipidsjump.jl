using Ipopt
using JuMP
using Plots

function kappa(s)
    0.5*exp.(-abs.(s))
end
function plotuv(x, u, v, p, keep=false)
    tails = u + v
    heads = circshift(u, -p) + circshift(v, p)
    if keep
        plot!(x, tails)
        plot!(x, heads)
    else
        plot(x, tails, label="tails")
        plot!(x, heads, label="heads")
    end
end
function lipidsjump()
    L = 30
    dx = 0.2 
    N = Int(2*L/dx + 1)
    x = -L:dx:L
    lipidlength = 5
    cmin = 1e-6
    p = Int(lipidlength/dx)
    c0 = 0.0625
    m = 2 
    d = m/dx + 2*N*c0
    alpha = 1

    K = zeros(N, N)
    for i = 1:N
        for j = 1:N
            k = kappa(dx*(i - j))
            if k > 1e-3 
                K[i, j] = k
            end
        end
    end




    model = Model(Ipopt.Optimizer)
    #model = Model(NLopt.Optimizer)
    #set_optimizer_attribute(model, "algorithm", :LD_SLSQP)
    @variables(model, begin
       u[1:N] >= cmin
       v[1:N] >= cmin
    end)
    @NLobjective(
        model,
        Min,
        sum(
            u[i]*log(u[i]) + v[i]*log(v[i]) + alpha*(1 - u[i] - v[i]) * 
            sum(K[i, j]*(u[j] + v[j]) for j = 1:N if K[i, j] > 0.0)
            for i in 1:N
        ) * dx,
    )

    @constraint(
        model, 
        ceq, 
        sum(u[i] + v[i] for i in 1:N) == d
    )
    @constraint(
        model,
        [i = 1:N], 
        u[i] + u[mod(i + p - 1, N) + 1] + v[i] + v[mod(i - p - 1, N) + 1] ≤ 1
    )
    optimize!(model)
    println("""
    termination_status = $(termination_status(model))
    primal_status      = $(primal_status(model))
    objective_value    = $(objective_value(model))
    """)

    return x, p, value.(u)[:], value.(v)[:]
end

x, p, u, v = lipidsjump()
# Because I'm not specifying an initial condition, it tends to produce a bilayer
# centred on the end of the domain (problem is invariant to cyclic shifts). so
# shift it back to the middle
s = length(u) ÷ 2
plotuv(x, circshift(u, s), circshift(v, s), p)