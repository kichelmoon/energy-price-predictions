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

# --- Train / Validation / Test Split (95% / 2.5% / 2.5%)
n_samples = length(y_raw)
train_end = Int(floor(0.95 * n_samples))
val_end   = Int(floor(0.975 * n_samples))

train_idx = 1:train_end
val_idx   = (train_end + 1):val_end
test_idx  = (val_end + 1):n_samples

X_train_raw, y_train_raw = X_raw[train_idx, :], y_raw[train_idx]
X_val_raw, y_val_raw     = X_raw[val_idx, :], y_raw[val_idx]
X_test_raw, y_test_raw   = X_raw[test_idx, :], y_raw[test_idx]

X_mean = mean(X_train_raw, dims=1)
X_std  = std(X_train_raw, dims=1)
X_std[X_std .== 0] .= 1.0

y_mean = mean(y_train_raw)
y_std  = std(y_train_raw)
y_std == 0 && (y_std = 1.0)

X_train = (X_train_raw .- X_mean) ./ X_std
y_train = (y_train_raw .- y_mean) ./ y_std

X_val = (X_val_raw .- X_mean) ./ X_std
y_val = (y_val_raw .- y_mean) ./ y_std

X_test = (X_test_raw .- X_mean) ./ X_std
y_test = (y_test_raw .- y_mean) ./ y_std


# --- 4. Ridge Regression ---
function fit_ridge(X, y, lambda)
    n_features = size(X, 2)
    I_reg = Matrix{Float64}(I, n_features, n_features)
    I_reg[1, 1] = 0.0 # Ignore intercept
    
    beta = (X' * X + lambda * I_reg) \ (X' * y)
    return beta
end

function ridge_regression(X_tr, y_tr, X_te, y_te, y_te_raw, lambda, feature_names)
    beta = fit_ridge(X_tr, y_tr, lambda)
    y_pred_norm = X_te * beta
    y_pred = (y_pred_norm .* y_std) .+ y_mean

    residuals = y_te_raw .- y_pred
    rmse = sqrt(mean(residuals .^ 2))
    mae = mean(abs.(residuals))
    r2 = 1.0 - (sum(residuals .^ 2) / sum((y_te_raw .- mean(y_te_raw)) .^ 2))

    println("\n--- Ridge Regression (Test Set), λ = $lambda ---")
    println("RMSE: ", round(rmse, digits=4))
    println("MAE:  ", round(mae, digits=4))
    println("R²:   ", round(r2, digits=4))

    println("\n--- Top 10 coefficients with intercept ---")
    coef_names = vcat([:Intercept], feature_names)
    for (name, coef) in zip(coef_names[1:min(10, end)], beta[1:min(10, end)])
        println("$(rpad(name, 25)): $(round(coef, digits=4))")
    end

    default(fontfamily="Arial", label="", titlefontsize=10, guidefontsize=9, tickfontsize=8)
    sub_df = df[test_idx, :]
    p1 = plot_predictions(sub_df, y_te_raw, y_pred, "Ridge Regression (Test)")
    p2 = plot_scatter(y_te_raw, y_pred)
    p3 = plot_residual_histogram(residuals)
    p4 = plot_top_features(feature_names, beta)    

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
    
    return y_pred, beta
end

ridge_prediction, ridge_beta = ridge_regression(X_train, y_train, X_test, y_test, y_test_raw, 5.0, feature_cols)

# --- Deep Learning
n_train_samples, n_features = size(X_train)
hidden_dim = 16

W1 = randn(Float32, n_features, hidden_dim) .* sqrt(2 / n_features)
b1 = zeros(Float32, 1, hidden_dim)
W2 = randn(Float32, hidden_dim, 1) .* sqrt(1 / hidden_dim)
b2 = zeros(Float32, 1, 1)

epochs = 100
learning_rate = 0.01
loss_history = Float64[]

for epoch in 1:epochs
    total_loss = 0.0
    indices = randperm(n_train_samples)
    
    for i in indices
        x_i = X_train[i:i, :]
        y_i = y_train[i:i, :]
        
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
    
    avg_epoch_loss = total_loss / n_train_samples
    push!(loss_history, avg_epoch_loss)

    if epoch % 10 == 0 || epoch == 1
        println("Epoch $epoch/$epochs - Loss (MSE): $(round(avg_epoch_loss, digits=6))")
    end
end

function predict(X_new_norm)
    z1 = X_new_norm * W1 .+ b1
    a1 = relu(z1)
    y_pred_norm = a1 * W2 .+ b2
    y_pred = (y_pred_norm .* y_std) .+ y_mean
    return y_pred
end

test_predictions = vec(predict(X_test))
residuals_dl = y_test_raw .- test_predictions

default(fontfamily="Arial", label="", titlefontsize=10, guidefontsize=9, tickfontsize=8)
sub_df_test = df[test_idx, :]

p1 = plot_predictions(sub_df_test, y_test_raw, test_predictions, "Deep Learning (Test)")
p2 = plot_scatter(y_test_raw, test_predictions)
p3 = plot_residual_histogram(residuals_dl)
p4 = plot_loss_convergence(epochs, loss_history)

dashboard_dl = plot(
    p1, p2, p3, p4,
    layout = (2, 2),
    size = (1200, 800),
    margin = 5Plots.mm,
    plot_title = "Dashboard for Deep Learning with α=$(round(learning_rate, digits=4))",
    plot_titlefontsize = 16
)

savefig(dashboard_dl, "plots/deep_learning_plots_epochs_$(Int(round(epochs))).png")
print("\n Saved Deep Learning dashboard!")