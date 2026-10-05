using Pkg
Pkg.add(["DataFrames", "CSV", "LinearAlgebra", "Statistics", "Plots"])

using CSV
using DataFrames
using LinearAlgebra
using Dates
using Statistics
using Plots

gr()

filename = "entsoe_hourly_data_de_lu.csv"
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
    I_reg[1, 1] = 0.0 # Den Intercept/Bias nicht regularisieren!
    
    beta = (X' * X + lambda * I_reg) \ (X' * y)
    return beta
end

lambda = 10.0
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


# Plot 1: Predicted vs Actual
p1 = plot(df.timestamp, y, label="Actual price", lw=2, color=:blue,
          title="Predicted prices and actual prices",
          xlabel="Date / Time", ylabel="EUR/MWh", legend=:topleft)
plot!(p1, df.timestamp, y_pred, label="Ridge Regression", lw=2, color=:orange, linestyle=:dash)

# Plot 2: Scatterplot target vs predicted
p2 = scatter(y, y_pred, label="Data points", alpha=0.7, color=:teal,
             title="Scatterplot: target vs predicted (R² = $(round(r2, digits=3)))",
             xlabel="Actual Price (EUR/MWh)", ylabel="Predicteed Price (EUR/MWh)")
plot!(p2, [minimum(y), maximum(y)], [minimum(y), maximum(y)], 
      label="(y = x)", color=:red, lw=2, linestyle=:dash)

# Plot 3: Histogram for residuals
p3 = histogram(residuals, bins=15, color=:purple, alpha=0.7, legend=false,
               title="Residual plot",
               xlabel="Error (y_true - y_pred)", ylabel="Frequency")
vline!(p3, [0], color=:black, lw=2, linestyle=:dash)

# Plot 4: Top features
feature_names = string.(feature_cols)
coef_importance = abs.(beta[2:end])

top_n = min(5, length(coef_importance))
sort_idx = sortperm(coef_importance, rev=true)[1:top_n]

top_names = feature_names[sort_idx]
top_values = coef_importance[sort_idx]

p4 = bar(
    top_values,
    orientation = :h,
    yticks = (1:top_n, top_names),
    color = :coral,
    legend = false,
    yflip = true,
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
dashboard = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 800), margin=5Plots.mm)
savefig(dashboard, "ridge_regression_plots.png")
print("\n Saved dashboard!")