using CSV, DataFrames, LinearAlgebra, Statistics

# Kriging模型结构体
struct KrigingModel
    X_train::Matrix{Float64}    # 训练输入
    y_train::Matrix{Float64}    # 训练输出  
    theta::Vector{Float64}      # RBF长度尺度参数（越大越重要）
    alpha::Matrix{Float64}      # 权重向量
    scaling_factors::Tuple      # 缩放因子信息
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
function preprocess_data(X, y)
    println("数据预处理: 调整不同特征到相近数量级...")
    
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
    
    return X_scaled, y_scaled, (X_scaling_factors, y_scaling_factors)
end

# 训练Kriging模型
function train_kriging(X, y)
    # 数据预处理：调整到相近数量级
    X_scaled, y_scaled, scaling_factors = preprocess_data(X, y)
    
    theta = get_physical_based_theta()
    
    println("\n使用基于物理意义的长度尺度参数:")
    feature_names = ["C_p_avg", "C_n_avg", "δ_SEI_nm", "c_f_Ah", "P_kW"]
    for i in 1:length(theta)
        importance = theta[i]  # theta越大越重要
        println("  $(feature_names[i]): θ = $(theta[i]) | 重要性: $(round(importance, digits=2))")
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
    println("构建核矩阵...")
    K = build_kernel_matrix(X_norm, theta)
    
    println("求解权重矩阵...")
    alpha = K \ y_norm
    
    model = KrigingModel(X_norm, y_norm, theta, alpha, scaling_factors)
    norm_params = (X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)
    
    return model, norm_params
end

# 预测函数（包含缩放和反缩放）
function predict(model::KrigingModel, norm_params, X_new)
    # 首先应用预处理缩放
    X_scaling_factors = model.scaling_factors[1]
    X_scaled = copy(X_new)
    
    for j in 1:size(X_new, 2)
        X_scaled[:, j] .= X_new[:, j] ./ X_scaling_factors[j]
    end
    
    # 然后应用标准化
    X_norm = (X_scaled .- norm_params.X_mean) ./ norm_params.X_std
    
    n_test = size(X_new, 1)
    n_outputs = size(model.y_train, 2)
    predictions_scaled = zeros(n_test, n_outputs)
    
    for i in 1:n_test
        k_star = [rbf_kernel(X_norm[i,:], model.X_train[j,:], model.theta) 
                 for j in 1:size(model.X_train,1)]
        
        y_pred_norm = k_star' * model.alpha
        predictions_scaled[i,:] = y_pred_norm .* norm_params.y_std .+ norm_params.y_mean
    end
    
    # 最后反缩放预测结果
    y_scaling_factors = model.scaling_factors[2]
    predictions = copy(predictions_scaled)
    
    for j in 1:size(predictions, 2)
        predictions[:, j] .= predictions_scaled[:, j] .* y_scaling_factors[j]
    end
    
    return predictions
end

# 分析特征重要性（theta越大越重要）
function analyze_feature_importance(model::KrigingModel, feature_names)
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

# 重新生成一周时间步长的训练数据
function generate_weekly_training_data()
    println("重新生成一周时间步长的训练数据...")
    
    # 加载现有数据
    df = CSV.read("D:\\vscode codes\\demo-script\\spm_training_data.csv", DataFrame)
    
    inputs = Matrix(df[:, [:C_p_avg, :C_n_avg, :δ_SEI_nm, :c_f_Ah, :P_kW]])
    outputs = Matrix(df[:, [:ΔC_p_avg, :ΔC_n_avg, :Δδ_SEI_pm, :Δc_f_μAh]])
    
    # 时间尺度放大到一周（保持原始单位）
    time_scale_factor = 168.0
    outputs .*= time_scale_factor
    
    return inputs, outputs
end

# 特征敏感性测试函数
function test_feature_sensitivity(predictor, base_state, base_power)
    """
    测试模型对不同特征的敏感性
    """
    println("\n特征敏感性测试:")
    println("基准状态: C_p=$(base_state[1]), C_n=$(base_state[2]), δ_SEI=$(base_state[3])nm, c_f=$(base_state[4])Ah")
    println("基准功率: $(base_power) kW")
    
    # 获取基准预测
    base_pred = predictor(base_state, base_power)
    
    println("\n变化对SEI增量的影响:")
    
    # 测试每个特征的变化
    variations = [
        ("C_p_avg +10%", [base_state[1]*1.1, base_state[2], base_state[3], base_state[4]], base_power),
        ("C_n_avg +10%", [base_state[1], base_state[2]*1.1, base_state[3], base_state[4]], base_power),
        ("δ_SEI +50%", [base_state[1], base_state[2], base_state[3]*1.5, base_state[4]], base_power),
        ("c_f +50%", [base_state[1], base_state[2], base_state[3], base_state[4]*1.5], base_power),
        ("P +50%", [base_state[1], base_state[2], base_state[3], base_state[4]], base_power*1.5),
    ]
    
    for (desc, state, power) in variations
        pred = predictor(state, power)
        delta_sei = pred[3] - base_pred[3]
        delta_cf = pred[4] - base_pred[4]
        
        println("  $desc:")
        println("    Δδ_SEI变化: $(round(delta_sei, digits=10)) pm")
        println("    Δc_f变化: $(round(delta_cf, digits=10)) μAh")
    end
end

# 主函数
function main()
    println("加载并调整SPM训练数据(一周时间步长)...")
    
    # 使用一周时间步长的数据
    inputs, outputs = generate_weekly_training_data()
    
    println("数据维度: $(size(inputs,1)) 样本, $(size(inputs,2)) 特征")
    
    # 验证训练数据
    validate_training_data(inputs, outputs)
    
    # 训练Kriging模型
    println("\n训练基于物理意义的Kriging代理模型...")
    model, norm_params = train_kriging(inputs, outputs)
    
    # 分析特征重要性
    feature_names = ["C_p_avg", "C_n_avg", "δ_SEI_nm", "c_f_Ah", "P_kW"]
    analyze_feature_importance(model, feature_names)
    
    return model, norm_params
end

# 使用函数
function use_kriging_model(model, norm_params)
    function predict_state_increment(state, power)
        input_vec = [state[1], state[2], state[3], state[4], power]
        input_mat = reshape(input_vec, 1, :)
        prediction = predict(model, norm_params, input_mat)
        return prediction[1,:]
    end
    return predict_state_increment
end

# 运行训练
println("开始基于物理意义的Kriging代理模型训练(一周时间步长)...")
model, norm_params = main()

# 创建预测函数
kriging_predictor = use_kriging_model(model, norm_params)

# 测试预测
println("\n" * "="^50)
println("模型预测测试(一周增量):")
println("="^50)

test_points = [
    ([6520.5, 11202.4, 15.2, 21.5], -23.1, "示例1-充电"),
    ([5899.5, 12676.4, 44.8, 7.5], -23.3, "示例2-充电"), 
    ([7659.0, 18277.6, 48.2, 43.7], 33.0, "示例3-放电"),
    ([10000.0, 10000.0, 1.0, 0.1], 25.0, "新电池-充电"),
    ([15000.0, 20000.0, 100.0, 20.0], 25.0, "老化电池-充电")
]

for (state, power, desc) in test_points
    pred = kriging_predictor(state, power)
    println("\n$desc:")
    println("  输入: C_p=$(state[1]), C_n=$(state[2]), δ_SEI=$(state[3])nm, c_f=$(state[4])Ah, P=$(power)kW")
    println("  预测增量（一周）:")
    println("    ΔC_p_avg: $(round(pred[1], digits=3))")
    println("    ΔC_n_avg: $(round(pred[2], digits=3))") 
    println("    Δδ_SEI: $(round(pred[3], digits=6)) pm")
    println("    Δc_f: $(round(pred[4], digits=6)) μAh")
end

# 特征敏感性测试
println("\n" * "="^50)
println("特征敏感性分析:")
println("="^50)
test_feature_sensitivity(kriging_predictor, [6520.5, 11202.4, 15.2, 21.5], -23.1)

println("\n基于物理意义的Kriging代理模型训练完成!")