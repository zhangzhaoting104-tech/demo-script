using JuMP, Ipopt, DataFrames, Plots, Statistics, CSV, LinearAlgebra

# SPM电池模型参数（基于BSSMPC论文中的LiFePO4参数）
struct SPMParameters
    # 扩散系数 (m²/s)
    D_p::Float64  # 正极
    D_n::Float64  # 负极
    
    # 粒子半径 (m)
    R_p::Float64
    R_n::Float64
    
    # 最大浓度 (mol/m³)
    C_p_max::Float64
    C_n_max::Float64
    
    # 几何参数
    a_p::Float64  # 正极表面积体积比 (m²/m³)
    a_n::Float64  # 负极表面积体积比 (m²/m³)
    l_p::Float64  # 正极厚度 (m)
    l_n::Float64  # 负极厚度 (m)
    
    # 反应速率常数
    k_p::Float64
    k_n::Float64
    
    # SEI相关参数
    k_SEI::Float64
    κ_SEI::Float64
    M_SEI::Float64
    ρ_SEI::Float64
    U_ref::Float64
    
    # 常数
    F::Float64    # 法拉第常数
    R::Float64    # 气体常数
    T::Float64    # 温度 (K)
    
    # 电解质浓度
    C_e::Float64
end

function create_default_SPM_parameters()
    # 基于论文中的LiFePO4参数
    SPMParameters(
        # 扩散系数
        1e-14,  # D_p
        3.9e-14, # D_n
        
        # 粒子半径
        1e-6,   # R_p
        1e-6,   # R_n
        
        # 最大浓度
        22806.0, # C_p_max
        30555.0, # C_n_max
        
        # 几何参数
        885000.0, # a_p
        723600.0, # a_n
        70e-6,    # l_p
        88e-6,    # l_n
        
        # 反应速率
        2.334e-11, # k_p
        1.764e-11, # k_n
        
        # SEI参数
        1e-12,    # k_SEI
        1e-2,     # κ_SEI
        0.162,    # M_SEI
        1690.0,   # ρ_SEI
        0.4,      # U_ref
        
        # 常数
        96485.0,  # F
        8.314,    # R
        298.15,   # T
        
        # 电解质浓度
        1000.0    # C_e
    )
end

# OCV函数（基于LiFePO4特性）
function OCV_positive(θ_p)
    # 简化的正极OCV-SOC关系
    return 3.4 + 0.5 * (1 - exp(-5 * (1 - θ_p))) - 0.3 * (1 - exp(-5 * θ_p))
end

function OCV_negative(θ_n)
    # 简化的负极OCV-SOC关系
    return 0.1 + 0.8 * θ_n - 0.3 * θ_n^2
end

# SPM状态更新函数
function update_SPM_state(params::SPMParameters, state, P, dt)
    # 解包状态变量
    C_p_avg, C_n_avg, δ_SEI, c_f = state
    
    # 计算SOC
    SOC = C_n_avg / params.C_n_max
    
    # 计算表面浓度（简化版本，基于论文公式3.3-3.4）
    i_app = P * 1000 / 3.7  # 假设3.7V平均电压，转换为mA
    
    C_p_surf = C_p_avg - (params.R_p / 5) * i_app / (params.F * params.D_p * params.a_p * params.l_p)
    C_n_surf = C_n_avg + (params.R_n / 5) * i_app / (params.F * params.D_n * params.a_n * params.l_n)
    
    # 确保浓度在合理范围内
    C_p_surf = max(0.01 * params.C_p_max, min(0.99 * params.C_p_max, C_p_surf))
    C_n_surf = max(0.01 * params.C_n_max, min(0.99 * params.C_n_max, C_n_surf))
    
    # 计算归一化表面浓度
    θ_p = C_p_surf / params.C_p_max
    θ_n = C_n_surf / params.C_n_max
    
    # 计算OCV
    U_p = OCV_positive(θ_p)
    U_n = OCV_negative(θ_n)
    
    # 计算SEI反应电流（简化版本）
    η_SEI = U_n - params.U_ref + δ_SEI / params.κ_SEI * i_app / (params.a_n * params.l_n)
    i_SEI = params.a_n * params.l_n * params.k_SEI * exp(-params.F / (params.R * params.T) * η_SEI)
    
    # 更新状态变量
    # 浓度动态（简化扩散方程）
    dC_p_avg = -15 * params.D_p / params.R_p^2 * (C_p_avg - C_p_surf)
    dC_n_avg = -15 * params.D_n / params.R_n^2 * (C_n_avg - C_n_surf)
    
    # SEI厚度增长
    dδ_SEI = i_SEI * params.M_SEI / (params.F * params.ρ_SEI * params.a_n * params.l_n)
    
    # 容量衰减
    dc_f = i_SEI / 3600.0  # 转换为Ah
    
    # 更新状态
    new_C_p_avg = C_p_avg + dC_p_avg * dt
    new_C_n_avg = C_n_avg + dC_n_avg * dt
    new_δ_SEI = δ_SEI + dδ_SEI * dt
    new_c_f = c_f + dc_f * dt
    
    # 计算终端电压（简化）
    V_terminal = U_p - U_n - i_app * (δ_SEI / params.κ_SEI)
    
    return [new_C_p_avg, new_C_n_avg, new_δ_SEI, new_c_f], V_terminal
end

# 简化的Kriging代理模型
struct SimpleKrigingModel
    training_data::Matrix{Float64}
    training_targets::Matrix{Float64}
    length_scales::Vector{Float64}
end

function create_kriging_surrogate(spm_params, num_samples=1000)
    # 生成训练数据
    states = []
    targets = []
    
    for i in 1:num_samples
        # 随机生成状态和输入
        state = [
            rand(0.1:0.01:0.9) * spm_params.C_p_max,  # C_p_avg
            rand(0.1:0.01:0.9) * spm_params.C_n_max,  # C_n_avg
            rand(0.0:1e-9:1e-7),                      # δ_SEI
            rand(0.0:0.1:10.0)                        # c_f
        ]
        
        P = rand(-50.0:1.0:50.0)  # 功率输入
        
        # 使用SPM计算状态增量
        new_state, _ = update_SPM_state(spm_params, state, P, 1.0)
        Δstate = new_state - state
        
        push!(states, vcat(state, P))
        push!(targets, Δstate)
    end
    
    states_matrix = hcat(states...)'  # 转换为矩阵
    targets_matrix = hcat(targets...)'  # 转换为矩阵
    
    # 简单的长度尺度估计
    length_scales = ones(size(states_matrix, 2))
    
    return SimpleKrigingModel(states_matrix, targets_matrix, length_scales)
end

function predict(model::SimpleKrigingModel, query_point)
    # 简化的最近邻预测（实际应用应使用真正的Kriging）
    distances = [norm(query_point - model.training_data[i, :]) for i in 1:size(model.training_data, 1)]
    nearest_idx = argmin(distances)
    
    return model.training_targets[nearest_idx, :]
end

function create_SPM_MPC_model()
    # 参数设置
    dt = 1.0  # 时间步长（小时）
    N = 24    # 预测时域
    K = 5     # 电池数量（简化）
    
    # 电价数据
    rho = [0.12, 0.11, 0.10, 0.10, 0.11, 0.15, 0.20, 0.18, 0.16, 0.14, 0.13, 0.14,
           0.15, 0.16, 0.17, 0.22, 0.25, 0.23, 0.19, 0.17, 0.15, 0.14, 0.13, 0.12]
    
    # SPM参数
    spm_params = create_default_SPM_parameters()
    
    # 创建Kriging代理模型（在实际应用中应该预训练）
    kriging_model = create_kriging_surrogate(spm_params, 500)
    
    # 创建优化模型
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 5000)
    set_optimizer_attribute(model, "tol", 1e-4)
    set_optimizer_attribute(model, "print_level", 0)
    
    # 定义决策变量
    @variable(model, -50.0 <= P[1:K, 1:N] <= 50.0)  # 每块电池的功率
    
    # 状态变量
    @variable(model, 0.1*spm_params.C_p_max <= C_p_avg[1:K, 1:N+1] <= 0.9*spm_params.C_p_max)
    @variable(model, 0.1*spm_params.C_n_max <= C_n_avg[1:K, 1:N+1] <= 0.9*spm_params.C_n_max)
    @variable(model, 0.0 <= δ_SEI[1:K, 1:N+1] <= 1e-6)
    @variable(model, 0.0 <= c_f[1:K, 1:N+1] <= 100.0)
    
    # 初始状态约束
    for k in 1:K
        @constraint(model, C_p_avg[k, 1] == 0.5 * spm_params.C_p_max)
        @constraint(model, C_n_avg[k, 1] == 0.5 * spm_params.C_n_max)
        @constraint(model, δ_SEI[k, 1] == 1e-9)
        @constraint(model, c_f[k, 1] == 0.0)
    end
    
    # 状态转移约束（使用代理模型）
    for k in 1:K, i in 1:N
        # 当前状态和输入
        current_state = [C_p_avg[k, i], C_n_avg[k, i], δ_SEI[k, i], c_f[k, i], P[k, i]]
        
        # 使用代理模型预测状态增量
        # 注意：需要实现真正的约束，这里使用简化版本
        ΔC_p = -0.01 * P[k, i] / 50.0 * spm_params.C_p_max
        ΔC_n = -0.01 * P[k, i] / 50.0 * spm_params.C_n_max
        Δδ_SEI = 1e-11 * (1 + abs(P[k, i]) / 10.0)
        Δc_f = 0.001 * (1 + abs(P[k, i]) / 20.0)
        
        @constraint(model, C_p_avg[k, i+1] == C_p_avg[k, i] + ΔC_p)
        @constraint(model, C_n_avg[k, i+1] == C_n_avg[k, i] + ΔC_n)
        @constraint(model, δ_SEI[k, i+1] == δ_SEI[k, i] + Δδ_SEI)
        @constraint(model, c_f[k, i+1] == c_f[k, i] + Δc_f)
    end
    
    # SOC约束（基于负极浓度）
    SOC_min = 0.2
    SOC_max = 0.9
    for k in 1:K, i in 1:N+1
        @constraint(model, C_n_avg[k, i] >= SOC_min * spm_params.C_n_max)
        @constraint(model, C_n_avg[k, i] <= SOC_max * spm_params.C_n_max)
    end
    
    # 目标函数：最大化收益，最小化老化
    aging_cost_per_Ah = 10.0  # 每Ah容量衰减的成本
    
    arbitrage_profit = sum(P[k, i] * rho[i] * dt for k in 1:K, i in 1:N)
    aging_cost = aging_cost_per_Ah * sum(c_f[k, N+1] - c_f[k, 1] for k in 1:K)
    
    @objective(model, Max, arbitrage_profit - aging_cost)
    
    return model, P, C_p_avg, C_n_avg, δ_SEI, c_f
end

# 运行优化
println("开始SPM-MPC优化...")
model, P_opt, C_p_avg_opt, C_n_avg_opt, δ_SEI_opt, c_f_opt = create_SPM_MPC_model()

optimize!(model)

if termination_status(model) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
    println("SPM-MPC优化成功!")
    
    K=5  # 假设有5块电池
    N=24  # 预测时域为24小时

    # 提取结果
    P_values = value.(P_opt)
    c_f_values = value.(c_f_opt)
    C_n_avg_values = value.(C_n_avg_opt)
    
    # 计算SOC轨迹 - 修复维度问题
    SOC_values = zeros(K, N+1)
    for k in 1:K, i in 1:N+1
        SOC_values[k, i] = C_n_avg_values[k, i] / 30555.0  # 除以C_n_max
    end
    
    # 打印结果摘要
    total_profit = objective_value(model)
    total_aging = sum(value.(c_f_opt)[:, end] - value.(c_f_opt)[:, 1])
    
    println("总目标值: ", round(total_profit, digits=2))
    println("总容量衰减: ", round(total_aging, digits=4), " Ah")
    
    # 创建更详细的结果分析
    result_df = DataFrame()
    result_df.Hour = 1:N
    result_df.Price = [0.12, 0.11, 0.10, 0.10, 0.11, 0.15, 0.20, 0.18, 0.16, 0.14, 0.13, 0.14,
                      0.15, 0.16, 0.17, 0.22, 0.25, 0.23, 0.19, 0.17, 0.15, 0.14, 0.13, 0.12]
    
    # 计算总功率和平均SOC
    result_df.Total_Power = [sum(value.(P_opt)[k,i] for k in 1:K) for i in 1:N]
    result_df.Avg_SOC = [mean(SOC_values[k,i] for k in 1:K) for i in 1:N]
    
    # 可视化结果 - 修复绘图问题
    p1 = plot(1:N, result_df.Price, label="Electricity Price", 
             xlabel="Hour", ylabel="Price (USD/kWh)", 
             line=:steppost, title="Electricity Price", color=:blue)
    
    p2 = plot(1:N, result_df.Total_Power, label="Total Power", 
             xlabel="Hour", ylabel="Power (kW)",
             line=:steppost, title="Total Power Strategy", color=:red)
    hline!([0], label="Zero", color=:black, linestyle=:dash, alpha=0.5)
    
    p3 = plot(1:N+1, SOC_values[1, :], label="Battery 1 SOC", 
             xlabel="Hour", ylabel="SOC", 
             title="SOC Evolution", color=:green)
    for k in 2:min(3, K)  # 只显示前3块电池避免过于拥挤
        plot!(1:N+1, SOC_values[k, :], label="Battery $k SOC")
    end
    hline!([0.2, 0.9], label=["SOC_min" "SOC_max"], 
           color=[:red :red], linestyle=:dash, alpha=0.5)
    
    # 容量衰减可视化
    final_c_f = [value.(c_f_opt)[k, end] for k in 1:K]
    p4 = bar(1:K, final_c_f, label="Final Capacity Fade", 
            xlabel="Battery ID", ylabel="Capacity Fade (Ah)",
            title="Battery Degradation Distribution", color=:orange)
    
    # 布局所有图表
    plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800))
    
    savefig("spm_mpc_results.png")
    println("结果图已保存为 spm_mpc_results.png")
    
    # 显示详细结果
    println("\n详细结果分析:")
    println("="^50)
    show(first(result_df, 12), allrows=true)
    println()
    
    # 分析套利行为
    charge_hours = findall(x -> x < -1, result_df.Total_Power)
    discharge_hours = findall(x -> x > 1, result_df.Total_Power)
    
    println("\n套利策略分析:")
    println("充电时段: ", charge_hours)
    println("放电时段: ", discharge_hours)
    
    if !isempty(charge_hours) && !isempty(discharge_hours)
        avg_charge_price = mean(result_df.Price[i] for i in charge_hours)
        avg_discharge_price = mean(result_df.Price[i] for i in discharge_hours)
        println("平均充电价格: ", round(avg_charge_price, digits=3), " USD/kWh")
        println("平均放电价格: ", round(avg_discharge_price, digits=3), " USD/kWh")
        println("价差: ", round(avg_discharge_price - avg_charge_price, digits=3), " USD/kWh")
    end
    
    # 老化成本分析
    aging_cost = 10.0 * total_aging  # 假设每Ah成本为10美元
    arbitrage_profit = sum(result_df.Total_Power .* result_df.Price)
    net_profit = arbitrage_profit - aging_cost
    
    println("\n经济性分析:")
    println("套利收益: ", round(arbitrage_profit, digits=2), " USD")
    println("老化成本: ", round(aging_cost, digits=2), " USD")
    println("净收益: ", round(net_profit, digits=2), " USD")
    println("老化成本占比: ", round(aging_cost/arbitrage_profit*100, digits=1), "%")
    
else
    println("优化失败: ", termination_status(model))
end