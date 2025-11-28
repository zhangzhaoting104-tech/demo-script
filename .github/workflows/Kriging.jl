using CSV, DataFrames, LinearAlgebra, Statistics

# Kriging模型结构体
struct KrigingModel
    X_train::Matrix{Float64}    # 训练输入
    y_train::Matrix{Float64}    # 训练输出  
    theta::Vector{Float64}      # RBF长度尺度参数
    alpha::Matrix{Float64}      # 权重向量
end

# RBF核函数
function rbf_kernel(x1, x2, theta)
    return exp(-sum(theta .* (x1 - x2).^2))
end

# 构建协方差矩阵
function build_kernel_matrix(X, theta)
    n = size(X, 1)
    K = zeros(n, n)
    for i in 1:n, j in 1:n
        K[i,j] = rbf_kernel(X[i,:], X[j,:], theta)
    end
    return K + 1e-10I
end

# 修正的长度尺度参数分配
function get_physical_based_theta()
    theta = zeros(5)
    theta[3] = 0.01  # δ_SEI: 最高敏感性
    theta[4] = 0.01  # c_f: 最高敏感性  
    theta[5] = 0.1   # P: 中等敏感性
    theta[1] = 1.0   # C_p_avg: 较低敏感性
    theta[2] = 1.0   # C_n_avg: 较低敏感性
    return theta
end

# 训练Kriging模型
function train_kriging(X, y)
    theta = get_physical_based_theta()
    
    println("使用基于物理意义的长度尺度参数:")
    feature_names = ["C_p_avg", "C_n_avg", "δ_SEI", "c_f", "P"]
    for i in 1:length(theta)
        println("  $(feature_names[i]): θ = $(theta[i])")
    end
    
    # 数据标准化
    X_mean = mean(X, dims=1)
    X_std = std(X, dims=1)
    X_std[X_std .== 0] .= 1.0
    X_norm = (X .- X_mean) ./ X_std
    
    y_mean = mean(y, dims=1)
    y_std = std(y, dims=1)
    y_std[y_std .== 0] .= 1.0
    y_norm = (y .- y_mean) ./ y_std
    
    # 构建核矩阵并求解权重
    println("构建核矩阵...")
    K = build_kernel_matrix(X_norm, theta)
    
    println("求解权重矩阵...")
    alpha = K \ y_norm
    
    model = KrigingModel(X_norm, y_norm, theta, alpha)
    norm_params = (X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)
    
    return model, norm_params
end

# 预测函数
function predict(model::KrigingModel, norm_params, X_new)
    X_norm = (X_new .- norm_params.X_mean) ./ norm_params.X_std
    
    n_test = size(X_new, 1)
    n_outputs = size(model.y_train, 2)
    predictions = zeros(n_test, n_outputs)
    
    for i in 1:n_test
        k_star = [rbf_kernel(X_norm[i,:], model.X_train[j,:], model.theta) 
                 for j in 1:size(model.X_train,1)]
        
        y_pred_norm = k_star' * model.alpha
        predictions[i,:] = y_pred_norm .* norm_params.y_std .+ norm_params.y_mean
    end
    
    return predictions
end

# 分析特征重要性
function analyze_feature_importance(model::KrigingModel, feature_names)
    println("\n特征重要性分析:")
    println("θ值越小表示特征越重要")
    println("-" ^ 50)
    
    for i in 1:length(model.theta)
        importance = 1.0 / model.theta[i]
        println("$(feature_names[i]): θ=$(model.theta[i]) | 重要性: $(round(importance, digits=2))")
    end
end

# 数据验证函数
function validate_training_data(inputs, outputs)
    println("\n训练数据验证:")
    println("输入数据范围:")
    feature_names = ["C_p_avg", "C_n_avg", "δ_SEI", "c_f", "P"]
    for i in 1:5
        println("  $(feature_names[i]): [$(minimum(inputs[:,i])), $(maximum(inputs[:,i]))]")
    end
    
    println("\n输出数据范围 (SEI增量):")
    println("  Δδ_SEI: [$(minimum(outputs[:,3])), $(maximum(outputs[:,3]))] m")
    println("  平均值: $(mean(outputs[:,3])) m")
    println("  非零SEI增量样本: $(sum(abs.(outputs[:,3]) .> 1e-15))/$(size(outputs,1))")
    
    significant_sei = outputs[:,3] .> 1e-18
    if sum(significant_sei) == 0
        println("⚠️ 警告: 训练数据中没有显著的SEI增量!")
    else
        println("✓ 训练数据包含显著的SEI增量")
        println("  显著样本数: $(sum(significant_sei))")
    end
end

# 重新生成一周时间步长的训练数据
function generate_weekly_training_data()
    println("重新生成一周时间步长的训练数据...")
    
    # 这里需要调用你的SPM数据生成函数，但将时间步长改为168小时
    # 由于你已经有SPM代码，这里假设调用相关函数
    # inputs, outputs = generate_SPM_data_with_timestep(168.0)  # 168小时 = 1周
    
    # 暂时先加载现有数据并调整时间尺度
    df = CSV.read("D:\\vscode codes\\demo-script\\spm_training_data.csv", DataFrame)
    
    inputs = Matrix(df[:, [:C_p_avg, :C_n_avg, :δ_SEI_nm, :c_f_Ah, :P_kW]])
    outputs = Matrix(df[:, [:ΔC_p_avg, :ΔC_n_avg, :Δδ_SEI_pm, :Δc_f_μAh]])
    
    # 单位转换
    inputs[:,3] ./= 1e9    # nm → m
    outputs[:,3] ./= 1e12  # pm → m  
    outputs[:,4] ./= 1e6   # μAh → Ah
    
    # 将1小时增量扩展到168小时（一周）
    # 注意：这不是简单的线性缩放，但作为初步近似
    time_scale_factor = 168.0
    outputs .*= time_scale_factor
    
    return inputs, outputs
end

# 主函数
function main()
    println("加载并调整SPM训练数据（一周时间步长）...")
    
    # 使用一周时间步长的数据
    inputs, outputs = generate_weekly_training_data()
    
    println("数据维度: $(size(inputs,1)) 样本, $(size(inputs,2)) 特征")
    
    # 验证训练数据
    validate_training_data(inputs, outputs)
    
    # 训练Kriging模型
    println("\n训练基于物理意义的Kriging代理模型...")
    model, norm_params = train_kriging(inputs, outputs)
    
    # 分析特征重要性
    feature_names = ["C_p_avg", "C_n_avg", "δ_SEI", "c_f", "P"]
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
println("开始基于物理意义的Kriging代理模型训练（一周时间步长）...")
model, norm_params = main()

# 创建预测函数
kriging_predictor = use_kriging_model(model, norm_params)

# 测试预测
println("\n" * "="^50)
println("模型预测测试（一周增量）:")
println("="^50)

test_points = [
    ([15000.0, 20000.0, 1e-9, 0.1], 25.0, "新电池-充电"),
    ([15000.0, 20000.0, 1e-7, 20.0], 25.0, "老化电池-充电"), 
    ([15000.0, 20000.0, 5e-8, 10.0], 50.0, "高功率充电")
]

for (state, power, desc) in test_points
    pred = kriging_predictor(state, power)
    println("\n$desc:")
    println("  输入: δ_SEI=$(state[3])m, c_f=$(state[4])Ah, P=$(power)kW")
    println("  预测增量（一周）:")
    println("    ΔC_p_avg: $(round(pred[1], digits=6))")
    println("    ΔC_n_avg: $(round(pred[2], digits=6))") 
    println("    Δδ_SEI: $(round(pred[3]*1e12, digits=6)) pm")
    println("    Δc_f: $(round(pred[4]*1e6, digits=6)) μAh")
end

println("\n基于物理意义的Kriging代理模型训练完成!")