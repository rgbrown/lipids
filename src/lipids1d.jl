module Lipids
export lipidbilayer, plotuv
using Ipopt
using JuMP
using Plots

function kappa_bap(s)
    0.5*exp.(-abs.(s))
end

function kappa_wht(s)
    -0.5*exp.(-abs.(s))
end

"""
    plotuv(x, u, v, p; keep=false, title="")

    Plot the head and tail distributions from the left and right oriented lipids u, v with length p indices
"""
function plotuv(x, u, v, p; keep=false, title="")
    tails = u + v
    heads = circshift(u, -p) + circshift(v, p)
    if keep
        plot!(x, tails, title)
        plot!(x, heads)
    else
        plot(x, tails, label="tails", title=title, ylim=(0, 1))
        plot!(x, heads, label="heads")
    end
end

"""
    lipidbilayer(L, dx, lipidlength, c0, m; 
    cmin=1e-5, alpha=1,sigma= 5, uc = 0, vc = 0, bilayermodel="bap")

Initialise problem 

`L`:                Periodic domain ``[-L, L]``
`dx`:               Domain spacing. Make sure dx divides L
`lipidlength`:      Length of lipid molecule (multiple of dx)
`c0`:               Background concentration
`m`:                Total lipid mass
`cmin`:              Minimum concentration of lipid. A small value instead
                    of 0 prevents log(0) ever being called
`sigma`             Standard deviation of initial populations
`uc`                Centre of initial u tail distribution
`vc`                Centre of initial v tail distribution
`bilayermodel`      "bap" or "wht"

Returns:
    `A`, `b`        Inequality constraints. A'x <= b
    `C`, `d`        Equality constraints. C'x = d
    `f`             Objective function
    `df`            Gradient function
    `y0`            Feasible initial condition
    `B`             Initial binding inequality constraints
"""
function lipidbilayer(L, dx, lipidlength, c0, m; 
    cmin=1e-5, alpha=1,sigma= 5, uc = 0, vc = 0, gamma=0.7, bilayermodel="bap")
    # Set up problem domain
    N = Int(2*L/dx + 1)
    x = -L:dx:L
    p = Int(lipidlength/dx)
    d = m/dx + 2*N*c0

    # Initial conditions, including normalisation step
    u0 = 1/(sigma*sqrt(2*pi))*exp.(-0.5*(x .- uc).^2/sigma^2) 
    v0 = 1/(sigma*sqrt(2*pi))*exp.(-0.5*(x .- vc).^2/sigma^2) 
    u0 = m/(2*sum(u0)*dx)*u0 .+ c0
    v0 = m/(2*sum(v0)*dx)*v0 .+ c0

    K = zeros(N, N)
    if bilayermodel == "bap"
        for i = 1:N
            for j = 1:N
                k = kappa_bap(dx*(i - j))
                if k > 1e-3 
                    K[i, j] = k
                end
            end
        end
    end
    if bilayermodel == "wht"
        for i = 1:N
            for j = 1:N
                k = kappa_wht(dx*(i - j))
                if k < -1e-3
                    K[i, j] = k
                end
            end
        end

    end

    model = Model(Ipopt.Optimizer)
    @variables(model, begin
       u[i = 1:N] >= cmin
       v[i = 1:N] >= cmin
    end)

    if bilayermodel == "bap"
        @NLobjective(
            model,
            Min,
            sum(
                u[i]*log(u[i]) + v[i]*log(v[i]) + alpha*(1 - u[i] - v[i]) * 
                sum(K[i, j]*(u[j] + v[j]) for j = 1:N if K[i, j] > 0.0)
                for i in 1:N
            ) * dx,
        )
    end
    if bilayermodel == "wht"
        @NLobjective(
            model,
            Min, 
            sum(
                u[i]*log(u[i]) + v[i]*log(v[i]) + 
                alpha*(1 - u[i] - v[i] - u[mod(i + p - 1, N) + 1] - v[mod(i - p - 1, N) + 1]) * 
                sum(K[i, j]*
                (1 - u[j] - v[j] - (1 - gamma)*u[mod(j + p - 1, N) + 1] - (1 - gamma)*v[mod(j - p - 1, N) + 1]) for j in 1:N if K[i, j] < 0.0)
                for i in 1:N
            ) * dx,
        )
    end
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

    return x, p, value.(u)[:], value.(v)[:], u0, v0
end
end
