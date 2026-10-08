using Plots

# Hex codes: Steel Blue, Terracotta, Forest Pine, Ochre, Slate Purple, Sand
palette_nordic = ["#3B5998", "#D96B43", "#2E6B56", "#E5A93C", "#6C5B7B", "#A89F91"]


function plot_predictions(df, y, y_pred, label_text)
    p1 = plot(df.timestamp, y, label="Actual price", lw=2, color=palette_nordic[1],
          title="Predicted prices and actual prices",
          xlabel="Date / Time", ylabel="EUR/MWh", legend=:topleft)
    plot!(p1, df.timestamp, y_pred, label=label_text, lw=2, color=palette_nordic[4], linestyle=:dash)
    return p1
end

function plot_scatter(y, y_pred)
    p2 = scatter(y, y_pred, label="Samples", alpha=0.7, color=palette_nordic[5],
             title="Scatterplot: Target vs Predicted",
             xlabel="Actual Price (EUR/MWh)", ylabel="Predicteed Price (EUR/MWh)")
    plot!(p2, [minimum(y), maximum(y)], [minimum(y), maximum(y)], label="1:1 Fit Line", color=:black, lw=2, linestyle=:dash)
    return p2
end

function plot_residual_histogram(residuals)
    p3 = histogram(residuals, bins=15, color=palette_nordic[3], alpha=0.7, legend=false,
               yscale = :log10,         # Logarithmic scale for y-axis
               ylims = (1, :auto),      # Deals with log(0) errors for empty bins
               title = "Residual plot (Log Scale)",
               xlabel = "Error (y_true - y_pred)", ylabel = "Frequency (log scale)")
    vline!(p3, [0], color=:black, lw=2, linestyle=:dash)
    return p3
end

function plot_top_features(feature_cols, beta)
    feature_names = string.(feature_cols)
    coef_importance = abs.(beta[2:end])

    top_n = min(10, length(coef_importance))
    sort_idx = sortperm(coef_importance, rev=true)[1:top_n]

    top_names = feature_names[sort_idx]
    top_values = coef_importance[sort_idx]

    p4 = bar(
        1:top_n,
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
        annotate!(p4, top_values[i] + (maximum(top_values) * 0.03), i, text(string(round(top_values[i], digits=2)), :left, 8, :black))
    end
    return p4
end

function plot_loss_convergence(epochs, loss_history)
    p4 = plot(1:epochs, loss_history, label="", color=palette_nordic[5], lw=2,
          title="SGD Training Loss Convergence",
          xlabel="Epoch", ylabel="MSE Loss")
    return p4
end