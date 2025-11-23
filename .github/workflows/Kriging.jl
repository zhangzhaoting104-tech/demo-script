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
    return K + 1e-8I  # 添加正则化
end

# 训练Kriging模型
function train_kriging(X, y; theta=nothing)
    n_features = size(X, 2)
    if theta === nothing
        theta = ones(n_features)  # 默认长度尺度
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
    K = build_kernel_matrix(X_norm, theta)
    alpha = K \ y_norm
    
    model = KrigingModel(X_norm, y_norm, theta, alpha)
    norm_params = (X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)
    
    return model, norm_params
end

# 预测函数
function predict(model::KrigingModel, norm_params, X_new)
    # 标准化输入
    X_norm = (X_new .- norm_params.X_mean) ./ norm_params.X_std
    
    n_test = size(X_new, 1)
    n_outputs = size(model.y_train, 2)
    predictions = zeros(n_test, n_outputs)
    
    for i in 1:n_test
        # 计算新点与训练点的核向量
        k_star = [rbf_kernel(X_norm[i,:], model.X_train[j,:], model.theta) for j in 1:size(model.X_train,1)]
        
        # Kriging预测
        y_pred_norm = k_star' * model.alpha
        
        # 反标准化
        predictions[i,:] = y_pred_norm .* norm_params.y_std .+ norm_params.y_mean
    end
    
    return predictions
end

# 主函数
function main()
    println("加载SPM训练数据...")
    
    # 加载数据
    df = CSV.read("D:\\vscode codes\\demo-script\\spm_training_data.csv", DataFrame)
    
    # 准备输入输出数据
    inputs = Matrix(df[:, [:C_p_avg, :C_n_avg, :δ_SEI_nm, :c_f_Ah, :P_kW]])
    outputs = Matrix(df[:, [:ΔC_p_avg, :ΔC_n_avg, :Δδ_SEI_pm, :Δc_f_μAh]])
    
    # 单位转换
    inputs[:,3] ./= 1e9   # nm -> m
    outputs[:,3] ./= 1e12 # pm -> m  
    outputs[:,4] ./= 1e6  # μAh -> Ah
    
    println("数据维度: $(size(inputs,1)) 样本, $(size(inputs,2)) 特征")
    
    # 训练Kriging模型
    println("训练Kriging代理模型...")
    model, norm_params = train_kriging(inputs, outputs)
    
    # 测试预测
    println("\n测试预测:")
    test_inputs = inputs[1:3,:]
    test_outputs = outputs[1:3,:]
    
    predictions = predict(model, norm_params, test_inputs)
    
    for i in 1:3
        println("样本 $i:")
        println("  真实: $(round.(test_outputs[i,:], digits=6))")
        println("  预测: $(round.(predictions[i,:], digits=6))")
    end
    
    return model, norm_params
end

# 使用函数
function use_kriging_model(model, norm_params)
    """
    使用训练好的Kriging模型进行预测
    """
    function predict_state_increment(state, power)
        """
        输入: state = [C_p_avg, C_n_avg, δ_SEI, c_f], power = 功率(kW)
        输出: [ΔC_p_avg, ΔC_n_avg, Δδ_SEI, Δc_f]
        """
        input_vec = [state[1], state[2], state[3], state[4], power]
        input_mat = reshape(input_vec, 1, :)
        
        prediction = predict(model, norm_params, input_mat)
        return prediction[1,:]
    end
    
    return predict_state_increment
end

# 运行训练
println("开始Kriging代理模型训练...")
model, norm_params = main()

# 创建预测函数
kriging_predictor = use_kriging_model(model, norm_params)

# 使用示例
println("\n" * "="^50)
println("Kriging代理模型使用示例:")
println("="^50)

# 示例1
state1 = [15000.0, 20000.0, 5e-8, 10.0]  # [C_p, C_n, δ_SEI, c_f]
power1 = 25.0
pred1 = kriging_predictor(state1, power1)
println("示例1 - 充电状态:")
println("  输入: C_p=$(state1[1]), C_n=$(state1[2]), δ_SEI=$(state1[3])m, P=$(power1)kW")
println("  预测增量: ΔC_p=$(round(pred1[1], digits=3)), ΔC_n=$(round(pred1[2], digits=3))")
println("            Δδ_SEI=$(round(pred1[3]*1e12, digits=6))pm, Δc_f=$(round(pred1[4]*1e6, digits=6))μAh")

# 示例2  
state2 = [10000.0, 15000.0, 1e-7, 5.0]
power2 = -30.0
pred2 = kriging_predictor(state2, power2)
println("\n示例2 - 放电状态:")
println("  输入: C_p=$(state2[1]), C_n=$(state2[2]), δ_SEI=$(state2[3])m, P=$(power2)kW")
println("  预测增量: ΔC_p=$(round(pred2[1], digits=3)), ΔC_n=$(round(pred2[2], digits=3))")
println("            Δδ_SEI=$(round(pred2[3]*1e12, digits=6))pm, Δc_f=$(round(pred2[4]*1e6, digits=6))μAh")

println("\nKriging代理模型就绪! 可以替代SPM进行快速预测。")

