module KrigingModelModule

using CSV, DataFrames, LinearAlgebra, Statistics

# 导出主要函数
export KrigingModel, MultiKrigingModel, train_kriging, predict, use_kriging_model, 
       load_training_data, create_and_train_kriging_model

# 单输出Kriging模型结构体
struct KrigingModel
    X_train::Matrix{Float64}    # 标准化后的训练输入
    y_train::Vector{Float64}    # 标准化后的训练输出（单输出）
    theta::Vector{Float64}      # RBF长度尺度参数
    alpha::Vector{Float64}      # 权重向量
    X_mean::Vector{Float64}     # 输入均值（用于标准化）
    X_std::Vector{Float64}      # 输入标准差（用于标准化）
    y_mean::Float64             # 输出均值（用于标准化）
    y_std::Float64              # 输出标准差（用于标准化）
    nugget::Float64             # 正则化参数
    output_scaling_factor::Float64  # 输出缩放因子
end

# 多输出Kriging模型包装器
struct MultiKrigingModel
    models::Vector{KrigingModel}  # 4个单输出模型
    feature_names::Vector{String}
    output_names::Vector{String}
end

# RBF核函数
function rbf_kernel(x1::AbstractVector{Float64}, x2::AbstractVector{Float64}, theta::Vector{Float64})
    kernel_value = 0.0
    for d in 1:length(x1)
        kernel_value += theta[d] * (x1[d] - x2[d])^2
    end
    return exp(-kernel_value)
end

# 改进的核矩阵构建（添加更强的正则化）
function build_kernel_matrix_with_nugget(X::Matrix{Float64}, theta::Vector{Float64}, output_idx::Int)
    n = size(X, 1)
    K = zeros(n, n)
    
    # 构建核矩阵
    for i in 1:n, j in 1:n
        K[i, j] = rbf_kernel(vec(X[i, :]), vec(X[j, :]), theta)
    end
    
    # 根据输出类型设置不同的nugget
    if output_idx <= 2
        # 前两个输出：较小的nugget，因为这些输出变化较大
        base_nugget = 1e-6
    else
        # 后两个输出：较大的nugget，因为它们是微小值，需要更平滑
        base_nugget = 1e-5
    end
    
    # 总是添加nugget避免完美拟合
    K += base_nugget * I
    
    return K, base_nugget
end

# 基于物理的theta优化
function optimize_theta_with_physics(X_norm::Matrix{Float64}, y_norm::Vector{Float64}, output_idx::Int)
    d = size(X_norm, 2)
    
    # 基于物理的重要性权重
    if output_idx == 1  # ΔC_p_avg
        # 主要受C_p, δ_SEI, c_f影响
        importance = [0.8, 0.2, 0.6, 0.6, 0.4]
    elseif output_idx == 2  # ΔC_n_avg
        # 主要受C_n, δ_SEI, c_f影响
        importance = [0.2, 0.8, 0.6, 0.6, 0.4]
    else  # 微小输出
        # δ_SEI和c_f是关键
        importance = [0.1, 0.1, 1.5, 1.5, 0.5]
    end
    
    # 基于特征范围和重要性计算theta
    theta = zeros(d)
    for j in 1:d
        # 计算特征的范围
        x_range = maximum(X_norm[:, j]) - minimum(X_norm[:, j])
        if x_range > 0
            # 基本theta与特征范围成反比
            base_theta = 1.0 / (x_range^2)
            # 乘以重要性权重
            theta[j] = base_theta * importance[j]
        else
            theta[j] = importance[j]
        end
    end
    
    # 确保theta在合理范围内
    theta = clamp.(theta, 0.05, 5.0)
    
    return theta
end

# 计算适当的输出缩放因子
function get_output_scaling_factor(output_idx::Int, y::Vector{Float64})
    if output_idx <= 2
        # ΔC_p_avg和ΔC_n_avg：不需要缩放
        return 1.0
    else
        # 计算输出的绝对值中位数
        y_abs = abs.(y)
        y_nonzero = y_abs[y_abs .> 0]
        
        if length(y_nonzero) > 0
            y_median = median(y_nonzero)
            # 目标缩放到数量级1
            scale = 1.0 / y_median
            # 限制缩放因子范围
            scale = clamp(scale, 1e4, 1e7)
            return scale
        else
            # 默认缩放因子
            return output_idx == 3 ? 1e6 : 1e5
        end
    end
end

# 训练单输出Kriging模型
function train_single_kriging(X::Matrix{Float64}, y::Vector{Float64}, 
                              output_idx::Int; verbose=true)
    if verbose
        println("\n训练第$(output_idx)个输出模型: $(output_idx <= 2 ? "ΔC" : "微小输出")")
    end
    
    # 输入标准化
    X_mean = vec(mean(X, dims=1))
    X_std = vec(std(X, dims=1))
    X_std[X_std .== 0] .= 1.0
    X_norm = (X .- X_mean') ./ X_std'
    
    # 输出缩放
    scaling_factor = get_output_scaling_factor(output_idx, y)
    y_scaled = y .* scaling_factor
    
    if verbose
        println("  输出缩放因子: $(round(scaling_factor, digits=2))")
        println("  缩放后输出范围: [$(round(minimum(y_scaled), digits=4)), $(round(maximum(y_scaled), digits=4))]")
    end
    
    # 输出标准化
    y_mean = mean(y_scaled)
    y_std = std(y_scaled)
    if y_std == 0
        y_std = 1.0
    end
    y_norm = (y_scaled .- y_mean) / y_std
    
    # 优化theta参数（基于物理）
    theta = optimize_theta_with_physics(X_norm, y_norm, output_idx)
    
    if verbose
        println("  优化的theta: $(round.(theta, digits=3))")
    end
    
    # 构建核矩阵
    K, nugget = build_kernel_matrix_with_nugget(X_norm, theta, output_idx)
    
    # 求解权重
    alpha = K \ y_norm
    
    # 计算训练误差
    y_pred_norm = K * alpha
    y_pred_scaled = y_pred_norm .* y_std .+ y_mean
    y_pred = y_pred_scaled ./ scaling_factor
    
    mae = mean(abs.(y_pred .- y))
    rmse = sqrt(mean((y_pred .- y).^2))
    
    if verbose
        cond_K = cond(K)
        println("  核矩阵条件数: $(round(cond_K, digits=2))")
        println("  nugget值: $(nugget)")
        println("  训练误差: MAE=$(round(mae, digits=10)), RMSE=$(round(rmse, digits=10))")
    end
    
    # 创建模型
    model = KrigingModel(X_norm, y_norm, theta, alpha, X_mean, X_std, 
                         y_mean, y_std, nugget, scaling_factor)
    
    return model
end

# 训练多输出Kriging模型
function train_kriging(X::Matrix{Float64}, y::Matrix{Float64}; verbose=true)
    if verbose
        println("="^60)
        println("训练Kriging模型（多输出）...")
        println("输入维度: $(size(X))")
        println("输出维度: $(size(y))")
    end
    
    n_outputs = size(y, 2)
    models = Vector{KrigingModel}(undef, n_outputs)
    
    for i in 1:n_outputs
        models[i] = train_single_kriging(X, y[:, i], i, verbose=verbose)
    end
    
    # 特征和输出名称
    feature_names = ["C_p_avg", "C_n_avg", "δ_SEI_nm", "c_f_Ah", "P_kW"]
    output_names = ["ΔC_p_avg", "ΔC_n_avg", "Δδ_SEI_pm", "Δc_f_μAh"]
    
    multi_model = MultiKrigingModel(models, feature_names, output_names)
    
    if verbose
        println("="^60)
        println("Kriging模型训练完成!")
    end
    
    return multi_model
end

# 单模型预测
function predict(model::KrigingModel, X_new::Matrix{Float64})
    n_test = size(X_new, 1)
    predictions = zeros(n_test)
    
    # 对输入进行标准化
    X_new_norm = (X_new .- model.X_mean') ./ model.X_std'
    
    # 对每个测试点进行预测
    for i in 1:n_test
        # 计算与所有训练样本的相似度
        k_star = zeros(size(model.X_train, 1))
        for j in 1:size(model.X_train, 1)
            k_star[j] = rbf_kernel(vec(X_new_norm[i, :]), vec(model.X_train[j, :]), model.theta)
        end
        
        # 预测（标准化空间）
        y_pred_norm = dot(k_star, model.alpha)
        
        # 反标准化和反缩放
        y_pred_scaled = y_pred_norm * model.y_std + model.y_mean
        predictions[i] = y_pred_scaled / model.output_scaling_factor
    end
    
    return predictions
end

# 多模型预测
function predict(multi_model::MultiKrigingModel, X_new::Matrix{Float64})
    n_test = size(X_new, 1)
    n_outputs = length(multi_model.models)
    predictions = zeros(n_test, n_outputs)
    
    for i in 1:n_outputs
        predictions[:, i] = predict(multi_model.models[i], X_new)
    end
    
    return predictions
end

# 针对单个输入的预测
function predict_single(multi_model::MultiKrigingModel, state::Vector{Float64}, power::Float64)
    input_vec = [state[1], state[2], state[3], state[4], power]
    input_mat = reshape(input_vec, 1, :)
    prediction = predict(multi_model, input_mat)
    return prediction[1, :]
end

# 创建预测器
function use_kriging_model(multi_model::MultiKrigingModel)
    function predict_state_increment(state::Vector{Float64}, power::Float64)
        return predict_single(multi_model, state, power)
    end
    return predict_state_increment
end

# 加载训练数据
function load_training_data(data_path="spm_training_data.csv")
    println("从 $data_path 加载训练数据...")
    
    df = CSV.read(data_path, DataFrame)
    
    println("CSV列名: $(names(df))")
    println("CSV维度: $(size(df,1)) 行 × $(size(df,2)) 列")
    
    inputs = Matrix{Float64}(df[:, 1:5])  # 前5列是输入
    outputs = Matrix{Float64}(df[:, 6:9]) # 后4列是输出
    
    println("数据维度: $(size(inputs,1)) 样本, $(size(inputs,2)) 特征")
    
    return inputs, outputs
end

# 数据验证
function validate_training_data(inputs::Matrix{Float64}, outputs::Matrix{Float64})
    println("\n原始训练数据验证:")
    println("输入数据范围:")
    feature_names = ["C_p_avg", "C_n_avg", "δ_SEI_nm", "c_f_Ah", "P_kW"]
    for i in 1:5
        min_val = minimum(inputs[:,i])
        max_val = maximum(inputs[:,i])
        mean_val = mean(inputs[:,i])
        println("  $(feature_names[i]): [$(round(min_val, digits=2)), $(round(max_val, digits=2))] | 均值: $(round(mean_val, digits=2))")
    end
    
    println("\n输出数据范围:")
    output_names = ["ΔC_p_avg", "ΔC_n_avg", "Δδ_SEI_pm", "Δc_f_μAh"]
    for i in 1:4
        min_val = minimum(outputs[:,i])
        max_val = maximum(outputs[:,i])
        mean_val = mean(outputs[:,i])
        println("  $(output_names[i]): [$(round(min_val, digits=10)), $(round(max_val, digits=10))] | 均值: $(round(mean_val, digits=10))")
    end
end

# 分析特征重要性
function analyze_feature_importance(multi_model::MultiKrigingModel)
    println("\n特征重要性分析:")
    println("θ值越大表示特征对相似度计算的影响越大")
    println("-" ^ 50)
    
    feature_names = multi_model.feature_names
    n_models = length(multi_model.models)
    
    for i in 1:n_models
        println("\n模型 $(multi_model.output_names[i]):")
        theta = multi_model.models[i].theta
        for j in 1:length(theta)
            println("  $(feature_names[j]): θ=$(round(theta[j], digits=4))")
        end
        
        total_theta = sum(theta)
        if total_theta > 0
            println("  相对重要性(百分比):")
            for j in 1:length(theta)
                relative_importance = theta[j] / total_theta * 100
                println("    $(feature_names[j]): $(round(relative_importance, digits=1))%")
            end
        end
    end
end

# 模型评估
function evaluate_model(multi_model::MultiKrigingModel, inputs::Matrix{Float64}, outputs::Matrix{Float64})
    println("\n模型评估:")
    println("="^60)
    
    # 预测所有训练数据
    predictions = predict(multi_model, inputs)
    
    # 计算每个输出的误差指标
    for i in 1:size(outputs, 2)
        actual = outputs[:, i]
        pred = predictions[:, i]
        
        # 计算各种误差指标
        mae = mean(abs.(pred .- actual))
        rmse = sqrt(mean((pred .- actual).^2))
        
        if i >= 3
            # 微小输出：显示绝对误差
            println("\n$(multi_model.output_names[i]):")
            println("  平均绝对误差: $(round(mae, digits=12))")
            println("  RMSE: $(round(rmse, digits=12))")
            
            # 计算相对误差（仅对非零值）
            non_zero_idx = findall(abs.(actual) .> 1e-12)
            if length(non_zero_idx) > 0
                mape = mean(abs.((pred[non_zero_idx] .- actual[non_zero_idx]) ./ actual[non_zero_idx])) * 100
                println("  平均相对误差（非零值）: $(round(mape, digits=2))%")
            end
        else
            # 常规输出
            mape = mean(abs.((pred .- actual) ./ (abs.(actual) .+ 1e-10))) * 100
            r2 = 1 - sum((pred .- actual).^2) / sum((actual .- mean(actual)).^2)
            
            println("\n$(multi_model.output_names[i]):")
            println("  MAE: $(round(mae, digits=6))")
            println("  RMSE: $(round(rmse, digits=6))")
            println("  MAPE: $(round(mape, digits=2))%")
            println("  R²: $(round(r2, digits=4))")
        end
    end
end

# 创建并训练模型的便捷函数
function create_and_train_kriging_model(data_path="spm_training_data.csv"; verbose=true)
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
    multi_model = train_kriging(inputs, outputs, verbose=verbose)
    
    # 分析特征重要性
    if verbose
        analyze_feature_importance(multi_model)
    end
    
    # 创建预测器
    predictor = use_kriging_model(multi_model)
    
    if verbose
        println("\n" * "="^60)
        println("Kriging模型训练完成!")
        println("="^60)
        
        # 模型评估
        evaluate_model(multi_model, inputs, outputs)
        
        # 测试预测
        println("\n模型验证测试（使用训练数据的前5个样本）:")
        for i in 1:min(5, size(inputs, 1))
            state = inputs[i, 1:4]
            power = inputs[i, 5]
            actual = outputs[i, :]
            pred = predictor(state, power)
            
            println("\n样本 $i:")
            println("  输入: C_p=$(round(state[1], digits=2)), C_n=$(round(state[2], digits=2)), δ_SEI=$(round(state[3], digits=2))nm, c_f=$(round(state[4], digits=2))Ah, P=$(round(power, digits=2))kW")
            println("  实际输出: ΔC_p=$(round(actual[1], digits=6)), ΔC_n=$(round(actual[2], digits=6)), Δδ_SEI=$(round(actual[3], digits=10)), Δc_f=$(round(actual[4], digits=10))")
            println("  预测输出: ΔC_p=$(round(pred[1], digits=6)), ΔC_n=$(round(pred[2], digits=6)), Δδ_SEI=$(round(pred[3], digits=10)), Δc_f=$(round(pred[4], digits=10))")
            
            # 计算相对误差
            errors = zeros(4)
            for j in 1:4
                if abs(actual[j]) > 1e-10
                    errors[j] = abs(pred[j] - actual[j]) / abs(actual[j]) * 100
                else
                    errors[j] = abs(pred[j] - actual[j]) * 100
                end
            end
            println("  相对误差(%): $(round(errors[1], digits=2))%, $(round(errors[2], digits=2))%, $(round(errors[3], digits=2))%, $(round(errors[4], digits=2))%")
        end
        
        # 额外测试点
        println("\n额外测试点预测:")
        test_points = [
            ([6520.5, 11202.4, 15.2, 21.5], -23.1, "示例1-充电"),
            ([5899.5, 12676.4, 44.8, 7.5], -23.3, "示例2-充电"), 
            ([7659.0, 18277.6, 48.2, 43.7], 33.0, "示例3-放电")
        ]
        
        for (state, power, desc) in test_points
            pred = predictor(state, power)
            println("\n$desc:")
            println("  输入: C_p=$(state[1]), C_n=$(state[2]), δ_SEI=$(round(state[3], digits=2))nm, c_f=$(round(state[4], digits=2))Ah, P=$(round(power, digits=2))kW")
            println("  预测增量:")
            println("    ΔC_p_avg: $(round(pred[1], digits=6))")
            println("    ΔC_n_avg: $(round(pred[2], digits=6))") 
            println("    Δδ_SEI: $(round(pred[3], digits=10)) pm")
            println("    Δc_f: $(round(pred[4], digits=10)) μAh")
        end
    end
    
    return multi_model, predictor
end

# 模块初始化代码
function __init__()
    println("KrigingModelModule 已加载")
end

end # module结束