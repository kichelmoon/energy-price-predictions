using Pkg
Pkg.add(["DataFrames", "CSV", "LinearAlgebra", "Statistics", "Plots"])

using CSV
using DataFrames
using LinearAlgebra
using Dates
using Statistics
using Plots

# Hex codes: Steel Blue, Terracotta, Forest Pine, Ochre, Slate Purple, Sand
palette_nordic = ["#3B5998", "#D96B43", "#2E6B56", "#E5A93C", "#6C5B7B", "#A89F91"]

gr()

filename = "data/entsoe_hourly_data_de_lu.csv"
df = CSV.read(filename, DataFrame)

rename!(df, 1 => :timestamp)
# Easier format to work with in plotting
clean_time_strings = first.(string.(df.timestamp), 19)
parsed_timestamps = DateTime.(clean_time_strings, dateformat"yyyy-mm-dd HH:MM:SS")

df.timestamp = parsed_timestamps
label_col = :energy_prices
feature_cols = filter(col -> col != label_col && col != :timestamp, propertynames(df))

y = Vector{Float64}(df[!, label_col])
X_raw = Matrix{Float64}(df[!, feature_cols])

# Z Score
means = mean(X_raw, dims=1)
stds = std(X_raw, dims=1)
stds[stds .== 0] .= 1.0 
X_scaled = (X_raw .- means) ./ stds
X = hcat(ones(size(X_scaled, 1)), X_scaled)

function fit_ridge(X, y, lambda)
    n_features = size(X, 2)
    I_reg = Matrix{Float64}(I, n_features, n_features)
    I_reg[1, 1] = 0.0 #Ignore intercept
    
    beta = (X' * X + lambda * I_reg) \ (X' * y) #Pseudoinverse goes brrrr
    return beta
end

lambda = 50.0
beta = fit_ridge(X, y, lambda)

y_pred = X * beta

residuals = y .- y_pred
rmse = sqrt(mean(residuals .^ 2))
mae = mean(abs.(residuals))
r2 = 1.0 - (sum(residuals .^ 2) / sum((y .- mean(y)) .^ 2))

println("\n--- Ridge Regression, λ = $lambda ---")
println("RMSE: ", round(rmse, digits=4))
println("MAE:  ", round(mae, digits=4))
println("R²:   ", round(r2, digits=4))

println("\n--- Top 10 coefficients with intercept ---")
coef_names = vcat([:Intercept], feature_cols)
for (name, coef) in zip(coef_names[1:min(10, end)], beta[1:min(10, end)])
    println("$(rpad(name, 25)): $(round(coef, digits=4))")
end

default(fontfamily="Arial", label="", titlefontsize=10, guidefontsize=9, tickfontsize=8)

# Plot 1: Predicted vs Actual
p1 = plot(df.timestamp, y, label="Actual price", lw=2, color=palette_nordic[1],
          title="Predicted prices and actual prices",
          xlabel="Date / Time", ylabel="EUR/MWh", legend=:topleft)
plot!(p1, df.timestamp, y_pred, label="Ridge Regression", lw=2, color=palette_nordic[4], linestyle=:dash)

# Plot 2: Scatterplot target vs predicted
p2 = scatter(y, y_pred, label="Samples", alpha=0.7, color=palette_nordic[5],
             title="Scatterplot: target vs predicted (R² = $(round(r2, digits=3)))",
             xlabel="Actual Price (EUR/MWh)", ylabel="Predicteed Price (EUR/MWh)")
plot!(p2, [minimum(y), maximum(y)], [minimum(y), maximum(y)], 
      label="1:1 Fit Line", color=:black, lw=2, linestyle=:dash)

# Plot 3: Histogram for residuals
p3 = histogram(residuals, bins=15, color=palette_nordic[3], alpha=0.7, legend=false,
               yscale = :log10,         # Logarithmic scale for y-axis
               ylims = (1, :auto),      # Prevents log(0) errors for empty bins
               title = "Residual plot (Log Scale)",
               xlabel = "Error (y_true - y_pred)", ylabel = "Frequency (log scale)")
vline!(p3, [0], color=:black, lw=2, linestyle=:dash)

# Plot 4: Top features
feature_names = string.(feature_cols)
coef_importance = abs.(beta[2:end])

top_n = min(10, length(coef_importance))
sort_idx = sortperm(coef_importance, rev=true)[1:top_n]

top_names = feature_names[sort_idx]
top_values = coef_importance[sort_idx]

p4 = bar(
    1:top_n,                     # Explicitly position each bar at unit intervals (1, 2, ..., top_n)
    top_values,
    orientation = :h,
    yticks = (1:top_n, top_names),
    color = palette_nordic[5],
    legend = false,
    yflip = true,
    ylims = (0.5, top_n + 0.5),   # Tightens vertical margins to eliminate blank padding
    xlims = (0, maximum(top_values) * 1.25),
    title = "Most predictive features",
    xlabel = "|β|",
    titlefontsize = 10,
    tickfontsize = 9,
    left_margin = 12Plots.mm
)

for i in 1:top_n
    annotate!(p4, top_values[i] + (maximum(top_values) * 0.03), i, 
              text(string(round(top_values[i], digits=2)), :left, 8, :black))
end

# Print Dashboard
dashboard = plot(
    p1, p2, p3, p4,
    layout = (2, 2),
    size = (1200, 800),
    margin = 5Plots.mm,
    plot_title = "Dashboard for Ridge Regression with λ=$(round(lambda, digits=2))",
    plot_titlefontsize = 16
)
savefig(dashboard, "plots/ridge_regression_plots_lambda_$(Int(round(lambda))).png")
print("\n Saved dashboard!")