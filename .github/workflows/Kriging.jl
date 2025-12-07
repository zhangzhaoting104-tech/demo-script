# Kriging_module.jl - 克里金模型模块
module KrigingModelModule1

using LinearAlgebra, Statistics, CSV, DataFrames, Optim

export create_and_train_kriging_model, predict_delta

struct KrigingPredictor
    X_train::Matrix{Float64}
    y_train::Matrix{Float64}
    theta::Vector{Float64}
    sigma::Float64
    mu_X::Vector{Float64}
    sigma_X::Vector{Float64}
    mu_y::Vector{Float64}
    sigma_y::Vector{Float64}
end

function normalize_data(X, y)
    mu_X = mean(X, dims=1)[:]
    sigma_X = std(X, dims=1)[:] .+ 1e-8
    mu_y = mean(y, dims=1)[:]
    sigma_y = std(y, dims=1)[:] .+ 1e-8
    
    X_norm = (X .- mu_X') ./ sigma_X'
    y_norm = (y .- mu_y') ./ sigma_y'
    
    return X_norm, y_norm, mu_X, sigma_X, mu_y, sigma_y
end

function correlation_matrix(X, theta)
    n = size(X, 1)
    R = Matrix{Float64}(undef, n, n)
    for i in 1:n
        for j in 1:n
            R[i, j] = exp(-sum(theta .* (X[i, :] .- X[j, :]).^2))
        end
        R[i, i] += 1e-8
    end
    return R
end

function log_likelihood(theta, X_norm, y_norm)
    n = size(X_norm, 1)
    R = correlation_matrix(X_norm, theta)
    R_chol = cholesky(R)
    
    # 计算负对数似然
    term1 = n * log(2π) + 2 * sum(log.(diag(R_chol.L)))
    term2 = sum((R_chol.L \ y_norm).^2)
    
    return 0.5 * (term1 + term2)
end

function create_and_train_kriging_model1(data_path::String="D:\\vscode codes\\demo-script\\spm_training_data.csv"; verbose::Bool=true)
    if verbose
        println("从 $data_path 加载训练数据...")
    end
    
    # 加载数据
    df = CSV.read(data_path, DataFrame)
    
    # 提取特征和目标
    X = Matrix(df[:, ["C_p_avg", "C_n_avg", "δ_SEI_nm", "c_f_Ah", "P_kW"]])
    y = Matrix(df[:, ["ΔC_p_avg", "ΔC_n_avg", "Δδ_SEI_pm", "Δc_f_μAh"]])
    y[:, 4] ./= 1000  # 将μAh转换为Ah
    
    if verbose
        println("数据维度: $(size(X, 1)) 样本, $(size(X, 2)) 特征")
    end
    
    # 归一化数据
    X_norm, y_norm, mu_X, sigma_X, mu_y, sigma_y = normalize_data(X, y)
    
    # 初始化theta
    init_theta = ones(size(X, 2))
    
    # 优化theta
    result = optimize(theta -> log_likelihood(theta, X_norm, y_norm), init_theta, LBFGS())
    theta_opt = Optim.minimizer(result)
    
    # 训练R矩阵
    R = correlation_matrix(X_norm, theta_opt)
    R_inv = inv(R)
    
    # 计算beta
    beta = R_inv * y_norm
    
    # 创建预测器
    predictor = KrigingPredictor(X, y, theta_opt, 1.0, mu_X, sigma_X, mu_y, sigma_y)
    
    # 创建预测函数
    function predict(x_new::Vector{Float64}, P::Float64)
        # 构造输入
        x_input = [x_new; P]
        
        # 归一化输入
        x_input_norm = (x_input .- predictor.mu_X) ./ predictor.sigma_X
        
        # 计算训练点的相关性
        r = Vector{Float64}(undef, size(predictor.X_train, 1))
        for i in 1:size(predictor.X_train, 1)
            r[i] = exp(-sum(predictor.theta .* (x_input_norm .- (predictor.X_train[i, :] .- predictor.mu_X)./predictor.sigma_X).^2))
        end
        
        # 归一化预测
        X_train_norm = (predictor.X_train .- predictor.mu_X') ./ predictor.sigma_X'
        y_train_norm = (predictor.y_train .- predictor.mu_y') ./ predictor.sigma_y'
        R_train = correlation_matrix(X_train_norm, predictor.theta)
        
        # Kriging预测
        y_norm_pred = r' * inv(R_train) * y_train_norm
        
        # 反归一化
        y_pred = y_norm_pred .* predictor.sigma_y .+ predictor.mu_y
        
        return y_pred[:]
    end
    
    return predictor, predict
end

end # 模块结束