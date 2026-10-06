using CSV
using DataFrames
using Random
using Statistics
using Plots

# Hex codes: Steel Blue, Terracotta, Forest Pine, Ochre, Slate Purple, Sand
palette_nordic = ["#3B5998", "#D96B43", "#2E6B56", "#E5A93C", "#6C5B7B", "#A89F91"]

# RELU activation function
relu(z) = max.(0, z)
d_relu(z) = Float32.(z .> 0)

df = CSV.read("data/entsoe_hourly_data_de_lu.csv", DataFrame)
rename!(df, 1 => :timestamp)

label_col = :energy_prices
feature_cols = filter(col -> col != label_col && col != :timestamp, propertynames(df))

y = Vector{Float64}(df[!, label_col])
X_raw = Matrix{Float64}(df[!, feature_cols])

# Z-score
X_mean = mean(X_raw, dims=1)
X_std  = std(X_raw, dims=1)
X = (X_raw .- X_mean) ./ X_std

y_mean = mean(y_raw)
y_std  = std(y_raw)
y = (y_raw .- y_mean) ./ y_std

n_samples, n_features = size(X)

hidden_dim = 16

# Initialize Layers
W1 = randn(Float32, n_features, hidden_dim) .* sqrt(2 / n_features)
b1 = zeros(Float32, 1, hidden_dim)
W2 = randn(Float32, hidden_dim, 1) .* sqrt(1 / hidden_dim)
b2 = zeros(Float32, 1, 1)

epochs = 1000
learning_rate = 0.001

loss_history = Float64[]

for epoch in 1:epochs
    total_loss = 0.0

    # SGD
    indices = randperm(n_samples)
    
    for i in indices
        x_i = X[i:i, :]
        y_i = y[i:i, :]
        
        z1 = x_i * W1 .+ b1
        a1 = relu(z1)
        z2 = a1 * W2 .+ b2
        y_pred = z2
        
        loss = 0.5 * (y_pred[1] - y_i[1])^2
        total_loss += loss
        
        dL_dz2 = y_pred .- y_i
        
        dW2 = a1' * dL_dz2
        db2 = dL_dz2
        
        dL_da1 = dL_dz2 * W2'
        dL_dz1 = dL_da1 .* d_relu(z1) 
        
        dW1 = x_i' * dL_dz1
        db1 = dL_dz1
        
        W2 .-= learning_rate .* dW2
        b2 .-= learning_rate .* db2
        W1 .-= learning_rate .* dW1
        b1 .-= learning_rate .* db1
    end
    
    push!(loss_history, total_loss / n_samples)

    if epoch % 10 == 0 || epoch == 1
        avg_loss = total_loss / n_samples
        println("Epoch $epoch/$epochs - Loss (MSE): $(round(avg_loss, digits=6))")
    end
end

function predict(X_new_raw)
    X_norm = (X_new_raw .- X_mean) ./ X_std
    
    z1 = X_norm * W1 .+ b1
    a1 = relu(z1)
    y_pred_norm = a1 * W2 .+ b2
    
    y_pred = (y_pred_norm .* y_std) .+ y_mean
    return y_pred
end

all_predictions = vec(predict(X_raw))
residuals = y_raw .- all_predictions

default(fontfamily="Arial", label="", titlefontsize=10, guidefontsize=9, tickfontsize=8)

# Plot 1: Predicted vs Actual
p1 = plot(df.timestamp, y_raw, label="Actual price", lw=2, color=palette_nordic[1],
          title="Predicted prices and actual prices",
          xlabel="Date / Time", ylabel="EUR/MWh", legend=:topleft)
plot!(p1, df.timestamp, all_predictions, label="Predicted price", color=palette_nordic[4], lw=1.0, linestyle=:dash)

# Plot 2: Scatterplot target vs predicted
p2 = scatter(y_raw, all_predictions, label="Samples", alpha=0.7, color=palette_nordic[5],
             title="Scatterplot: target vs predicted",
             xlabel="Actual Price (EUR/MWh)", ylabel="Predicteed Price (EUR/MWh)")
plot!(p2, [minimum(y_raw), maximum(y_raw)], [minimum(y_raw), maximum(y_raw)],
      label="1:1 Fit Line", color=:black, linestyle=:dash)

# Plot 3: Histogram for residuals
p3 = histogram(residuals, bins=15, color=palette_nordic[3], alpha=0.7, legend=false,
               yscale = :log10,         # Logarithmic scale for y-axis
               ylims = (1, :auto),      # Prevents log(0) errors for empty bins
               title = "Residual plot (Log Scale)",
               xlabel = "Error (y_true - y_pred)", ylabel = "Frequency (log scale)")
vline!(p3, [0], color=:black, lw=2, linestyle=:dash)

# Plot 4: Training Loss Convergence
p4 = plot(1:epochs, loss_history, label="", color=palette_nordic[5], lw=2,
          title="SGD Training Loss Convergence",
          xlabel="Epoch", ylabel="MSE Loss")

dashboard = plot(
    p1, p2, p3, p4,
    layout = (2, 2),
    size = (1200, 800),
    margin = 5Plots.mm,
    plot_title = "Dashboard for Deep Learning with α=$(round(learning_rate, digits=4))",
    plot_titlefontsize = 16
)
savefig(dashboard, "plots/deep_learning_plots_epochs_$(Int(round(epochs))).png")
print("\n Saved dashboard!")