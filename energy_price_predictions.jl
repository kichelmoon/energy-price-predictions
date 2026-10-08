using Pkg
Pkg.add(["DataFrames", "CSV", "LinearAlgebra", "Statistics", "Plots"])

using CSV
using DataFrames
using LinearAlgebra
using Dates
using Statistics
using Plots
using Random

include("plots.jl")

# RELU activation function for deep learning
relu(z) = max.(0, z)
d_relu(z) = Float32.(z .> 0)

#---Raw data to workable data frame, including Z-Score Normalization
df = CSV.read("data/entsoe_hourly_data_de_lu.csv", DataFrame)
rename!(df, 1 => :timestamp)
# Easier format to work with in plotting
clean_time_strings = first.(string.(df.timestamp), 19)
parsed_timestamps = DateTime.(clean_time_strings, dateformat"yyyy-mm-dd HH:MM:SS")
df.timestamp = parsed_timestamps

label_col = :energy_prices
feature_cols = filter(col -> col != label_col && col != :timestamp, propertynames(df))

y_raw = Vector{Float64}(df[!, label_col])
X_raw = Matrix{Float64}(df[!, feature_cols])

# Z-score
X_mean = mean(X_raw, dims=1)
X_std  = std(X_raw, dims=1)
X = (X_raw .- X_mean) ./ X_std

y_mean = mean(y_raw)
y_std  = std(y_raw)
y = (y_raw .- y_mean) ./ y_std

#--- Ridge Regression
function fit_ridge(X, y, lambda)
    n_features = size(X, 2)
    I_reg = Matrix{Float64}(I, n_features, n_features)
    I_reg[1, 1] = 0.0 #Ignore intercept
    
    beta = (X' * X + lambda * I_reg) \ (X' * y) #Pseudoinverse goes brrrr
    return beta
end

function ridge_regression(X, y, lambda)
    beta = fit_ridge(X, y, lambda)
    y_pred_norm = X * beta
    y_pred = (y_pred_norm .* y_std) .+ y_mean

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
    p1 = plot_predictions(df, y_raw, y_pred, "Ridge Regression")
    p2 = plot_scatter(y_raw, y_pred)
    p3 = plot_residual_histogram(residuals)
    p4 = plot_top_features(feature_cols, beta)    

    dashboard = plot(
        p1, p2, p3, p4,
        layout = (2, 2),
        size = (1200, 800),
        margin = 5Plots.mm,
        plot_title = "Dashboard for Ridge Regression with λ=$(round(lambda, digits=2))",
        plot_titlefontsize = 16
    )
    savefig(dashboard, "plots/ridge_regression_plots_lambda_$(Int(round(lambda))).png")
    print("\n Saved Ridge Regression dashboard!")
    
    return y_pred
end

ridge_prediction, ridge_beta = ridge_regression(X, y, 5.0)

#--- Deep Learning
n_samples, n_features = size(X)

hidden_dim = 16

# Initialize Layers
W1 = randn(Float32, n_features, hidden_dim) .* sqrt(2 / n_features)
b1 = zeros(Float32, 1, hidden_dim)
W2 = randn(Float32, hidden_dim, 1) .* sqrt(1 / hidden_dim)
b2 = zeros(Float32, 1, 1)

epochs = 100
learning_rate = 0.01

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

p1 = plot_predictions(df, y_raw, all_predictions, "Deep Learning")
p2 = plot_scatter(y_raw, all_predictions)
p3 = plot_residual_histogram(residuals)
p4 = plot_loss_convergence(epochs, loss_history)

dashboard = plot(
    p1, p2, p3, p4,
    layout = (2, 2),
    size = (1200, 800),
    margin = 5Plots.mm,
    plot_title = "Dashboard for Deep Learning with α=$(round(learning_rate, digits=4))",
    plot_titlefontsize = 16
)
savefig(dashboard, "plots/deep_learning_plots_epochs_$(Int(round(epochs))).png")
print("\n Saved Deep Learning dashboard!")