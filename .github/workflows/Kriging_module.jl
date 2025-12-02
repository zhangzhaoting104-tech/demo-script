module KrigingModelModule

using CSV, DataFrames, LinearAlgebra, Statistics

# 导出主要函数
export KrigingModel, train_kriging, predict, use_kriging_model, load_training_data,
       create_and_train_kriging_model

# Kriging模型结构体
struct KrigingModel
    X_train::Matrix{Float64}    # 训练输入
    y_train::Matrix{Float64}    # 训练输出  
    theta::Vector{Float64}      # RBF长度尺度参数（越大越重要）
    alpha::Matrix{Float64}      # 权重向量
    scaling_factors::Tuple      # 缩放因子信息
    norm_params::NamedTuple     # 标准化参数
end

# 正确的RBF核函数（根据公式3.17）
function rbf_kernel(x1, x2, theta)
    # ∏_{d=1}^{D} exp(-θ_d |x_d - x'_d|^2)
    kernel_value = 1.0
    for d in 1:length(x1)
        kernel_value *= exp(-theta[d] * (x1[d] - x2[d])^2)
    end
    return kernel_value
end

# 构建协方差矩阵
function build_kernel_matrix(X, theta)
    n = size(X, 1)
    K = zeros(n, n)
    for i in 1:n, j in 1:n
        K[i,j] = rbf_kernel(X[i,:], X[j,:], theta)
    end
    return K + 1e-8I  # 添加小的正则项保证数值稳定性
end

# 修正的长度尺度参数分配（theta越大越重要）
function get_physical_based_theta()
    theta = zeros(5)
    # 根据电池老化物理机制设置theta
    # theta越大表示该特征对相似度计算的影响越大（越重要）
    theta[1] = 1    # C_p_avg: 较低敏感性
    theta[2] = 1    # C_n_avg: 较低敏感性
    theta[3] = 1.5   # δ_SEI: 最高敏感性 - SEI厚度对老化影响最大
    theta[4] = 1.5   # c_f: 最高敏感性 - 容量衰减是关键老化指标
    theta[5] = 1.2    # P: 中等敏感性 - 功率影响老化速率
    return theta
end

# 数据预处理：将不同数量级的数据调整到相近范围
function preprocess_data(X, y; verbose=true)
    if verbose
        println("数据预处理: 调整不同特征到相近数量级...")
    end
    
    # 输入特征缩放因子（基于特征典型值）
    X_scaling_factors = [
        1000.0,  # C_p_avg: 从千级别缩放到10级别
        1000.0,  # C_n_avg: 从千级别缩放到10级别  
        1.0,     # δ_SEI_nm: 保持nm单位（数值在合理范围）
        10.0,    # c_f_Ah: 从10级别缩放到1级别
        10.0     # P_kW: 从10级别缩放到1级别
    ]
    
    # 输出特征缩放因子
    y_scaling_factors = [
        10.0,    # ΔC_p_avg: 从10级别缩放到1级别
        10.0,    # ΔC_n_avg: 从10级别缩放到1级别
        1.0,     # Δδ_SEI_pm: 保持pm单位（数值在合理范围）
        1.0      # Δc_f_μAh: 保持μAh单位（数值在合理范围）
    ]
    
    # 应用缩放
    X_scaled = copy(X)
    y_scaled = copy(y)
    
    for j in 1:size(X, 2)
        X_scaled[:, j] .= X[:, j] ./ X_scaling_factors[j]
    end
    
    for j in 1:size(y, 2)
        y_scaled[:, j] .= y[:, j] ./ y_scaling_factors[j]
    end
    
    if verbose
        println("缩放后的输入数据范围:")
        feature_names = ["C_p_avg", "C_n_avg", "δ_SEI_nm", "c_f_Ah", "P_kW"]
        for i in 1:5
            min_val = minimum(X_scaled[:, i])
            max_val = maximum(X_scaled[:, i])
            mean_val = mean(X_scaled[:, i])
            println("  $(feature_names[i]): [$(round(min_val, digits=3)), $(round(max_val, digits=3))] | 均值: $(round(mean_val, digits=3))")
        end
        
        println("缩放后的输出数据范围:")
        output_names = ["ΔC_p_avg", "ΔC_n_avg", "Δδ_SEI_pm", "Δc_f_μAh"]
        for i in 1:4
            min_val = minimum(y_scaled[:, i])
            max_val = maximum(y_scaled[:, i])
            mean_val = mean(y_scaled[:, i])
            println("  $(output_names[i]): [$(round(min_val, digits=3)), $(round(max_val, digits=3))] | 均值: $(round(mean_val, digits=3))")
        end
    end
    
    return X_scaled, y_scaled, (X_scaling_factors, y_scaling_factors)
end

# 训练Kriging模型
function train_kriging(X, y; verbose=true)
    # 数据预处理：调整到相近数量级
    if verbose
        println("训练Kriging模型...")
    end
    X_scaled, y_scaled, scaling_factors = preprocess_data(X, y, verbose=verbose)
    
    theta = get_physical_based_theta()
    
    if verbose
        println("\n使用基于物理意义的长度尺度参数:")
        feature_names = ["C_p_avg", "C_n_avg", "δ_SEI_nm", "c_f_Ah", "P_kW"]
        for i in 1:length(theta)
            importance = theta[i]  # theta越大越重要
            println("  $(feature_names[i]): θ = $(theta[i]) | 重要性: $(round(importance, digits=2))")
        end
    end
    
    # 数据标准化（在缩放后的数据上进行）
    X_mean = mean(X_scaled, dims=1)
    X_std = std(X_scaled, dims=1)
    X_std[X_std .== 0] .= 1.0
    X_norm = (X_scaled .- X_mean) ./ X_std
    
    y_mean = mean(y_scaled, dims=1)
    y_std = std(y_scaled, dims=1)
    y_std[y_std .== 0] .= 1.0
    y_norm = (y_scaled .- y_mean) ./ y_std
    
    # 构建核矩阵并求解权重
    if verbose
        println("构建核矩阵...")
    end
    K = build_kernel_matrix(X_norm, theta)
    
    if verbose
        println("求解权重矩阵...")
    end
    alpha = K \ y_norm
    
    norm_params = (X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)
    model = KrigingModel(X_norm, y_norm, theta, alpha, scaling_factors, norm_params)
    
    if verbose
        println("Kriging模型训练完成!")
    end
    
    return model
end

# 预测函数（包含缩放和反缩放）
function predict(model::KrigingModel, X_new)
    # 首先应用预处理缩放
    X_scaling_factors = model.scaling_factors[1]
    X_scaled = copy(X_new)
    
    for j in 1:size(X_new, 2)
        X_scaled[:, j] .= X_new[:, j] ./ X_scaling_factors[j]
    end
    
    # 然后应用标准化
    X_norm = (X_scaled .- model.norm_params.X_mean) ./ model.norm_params.X_std
    
    n_test = size(X_new, 1)
    n_outputs = size(model.y_train, 2)
    predictions_scaled = zeros(n_test, n_outputs)
    
    for i in 1:n_test
        k_star = [rbf_kernel(X_norm[i,:], model.X_train[j,:], model.theta) 
                 for j in 1:size(model.X_train,1)]
        
        y_pred_norm = k_star' * model.alpha
        predictions_scaled[i,:] = y_pred_norm .* model.norm_params.y_std .+ model.norm_params.y_mean
    end
    
    # 最后反缩放预测结果
    y_scaling_factors = model.scaling_factors[2]
    predictions = copy(predictions_scaled)
    
    for j in 1:size(predictions, 2)
        predictions[:, j] .= predictions_scaled[:, j] .* y_scaling_factors[j]
    end
    
    return predictions
end

# 方便的预测函数（针对单个输入）
function predict_single(model::KrigingModel, state, power)
    """
    针对单个状态和功率进行预测
    state: [C_p_avg, C_n_avg, δ_SEI, c_f]
    power: 功率值 (kW)
    """
    input_vec = [state[1], state[2], state[3], state[4], power]
    input_mat = reshape(input_vec, 1, :)
    prediction = predict(model, input_mat)
    return prediction[1, :]
end

# 使用函数（创建预测器）
function use_kriging_model(model::KrigingModel)
    """
    创建一个方便的Kriging预测器函数
    返回的函数接受state和power，返回状态增量
    """
    function predict_state_increment(state, power)
        return predict_single(model, state, power)
    end
    return predict_state_increment
end

# 加载训练数据
function load_training_data(data_path="D:\\vscode codes\\demo-script\\spm_training_data.csv")
    """
    从CSV文件加载训练数据
    """
    println("从 $data_path 加载训练数据...")
    
    # 加载现有数据
    df = CSV.read(data_path, DataFrame)
    
    inputs = Matrix(df[:, [:C_p_avg, :C_n_avg, :δ_SEI_nm, :c_f_Ah, :P_kW]])
    outputs = Matrix(df[:, [:ΔC_p_avg, :ΔC_n_avg, :Δδ_SEI_pm, :Δc_f_μAh]])
    
    # 时间尺度放大到一周（保持原始单位）
    time_scale_factor = 168.0
    outputs .*= time_scale_factor
    
    println("数据维度: $(size(inputs,1)) 样本, $(size(inputs,2)) 特征")
    
    return inputs, outputs
end

# 数据验证函数
function validate_training_data(inputs, outputs)
    println("\n原始训练数据验证:")
    println("输入数据范围:")
    feature_names = ["C_p_avg", "C_n_avg", "δ_SEI_nm", "c_f_Ah", "P_kW"]
    for i in 1:5
        min_val = minimum(inputs[:,i])
        max_val = maximum(inputs[:,i])
        mean_val = mean(inputs[:,i])
        println("  $(feature_names[i]): [$(round(min_val, digits=2)), $(round(max_val, digits=2))] | 均值: $(round(mean_val, digits=2))")
    end
    
    println("\n输出数据范围(一周增量):")
    output_names = ["ΔC_p_avg", "ΔC_n_avg", "Δδ_SEI_pm", "Δc_f_μAh"]
    for i in 1:4
        min_val = minimum(outputs[:,i])
        max_val = maximum(outputs[:,i])
        mean_val = mean(outputs[:,i])
        println("  $(output_names[i]): [$(round(min_val, digits=6)), $(round(max_val, digits=6))] | 均值: $(round(mean_val, digits=6))")
    end
end

# 分析特征重要性
function analyze_feature_importance(model::KrigingModel, feature_names=["C_p_avg", "C_n_avg", "δ_SEI_nm", "c_f_Ah", "P_kW"])
    println("\n特征重要性分析:")
    println("θ值越大表示特征越重要")
    println("-" ^ 50)
    
    for i in 1:length(model.theta)
        importance = model.theta[i]
        println("$(feature_names[i]): θ=$(model.theta[i]) | 重要性: $(round(importance, digits=2))")
    end
    
    # 计算相对重要性百分比
    total_theta = sum(model.theta)
    println("\n相对重要性(百分比):")
    for i in 1:length(model.theta)
        relative_importance = model.theta[i] / total_theta * 100
        println("  $(feature_names[i]): $(round(relative_importance, digits=1))%")
    end
end

# 创建并训练模型的便捷函数
function create_and_train_kriging_model(data_path="spm_training_data.csv"; verbose=true)
    """
    一站式函数：加载数据并训练Kriging模型
    返回模型和预测器
    """
    if verbose
        println("="^60)
        println("开始训练Kriging代理模型")
        println("="^60)
    end
    
    # 加载数据
    inputs, outputs = load_training_data(data_path)
    
    # 验证数据
    if verbose
        validate_training_data(inputs, outputs)
    end
    
    # 训练模型
    model = train_kriging(inputs, outputs, verbose=verbose)
    
    # 分析特征重要性
    if verbose
        analyze_feature_importance(model)
    end
    
    # 创建预测器
    predictor = use_kriging_model(model)
    
    if verbose
        println("\n" * "="^60)
        println("Kriging模型训练完成!")
        println("="^60)
        
        # 测试预测
        println("\n模型预测测试(一周增量):")
        test_points = [
            ([6520.5, 11202.4, 15.2, 21.5], -23.1, "示例1-充电"),
            ([5899.5, 12676.4, 44.8, 7.5], -23.3, "示例2-充电"), 
            ([7659.0, 18277.6, 48.2, 43.7], 33.0, "示例3-放电")
        ]
        
        for (state, power, desc) in test_points
            pred = predictor(state, power)
            println("\n$desc:")
            println("  输入: C_p=$(state[1]), C_n=$(state[2]), δ_SEI=$(state[3])nm, c_f=$(state[4])Ah, P=$(power)kW")
            println("  预测增量（一周）:")
            println("    ΔC_p_avg: $(round(pred[1], digits=3))")
            println("    ΔC_n_avg: $(round(pred[2], digits=3))") 
            println("    Δδ_SEI: $(round(pred[3], digits=6)) pm")
            println("    Δc_f: $(round(pred[4], digits=6)) μAh")
        end
    end
    
    return model, predictor
end

# 保存模型到文件（使用CSV格式简化版本）
function save_model_simple(model::KrigingModel, base_path="kriging_model")
    """
    简化版模型保存，将模型参数保存为CSV文件
    """
    # 保存模型参数
    CSV.write("$(base_path)_params.csv", DataFrame(
        parameter = ["theta_1", "theta_2", "theta_3", "theta_4", "theta_5"],
        value = model.theta
    ))
    
    # 保存缩放因子
    CSV.write("$(base_path)_scaling.csv", DataFrame(
        X_scaling_factors = model.scaling_factors[1],
        y_scaling_factors = model.scaling_factors[2]
    ))
    
    println("模型参数已保存到 $(base_path)_*.csv 文件")
end

# 模块初始化代码
function __init__()
    println("KrigingModelModule 已加载")
end

end # module结束