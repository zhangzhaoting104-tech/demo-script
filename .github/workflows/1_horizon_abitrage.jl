using JuMP, Ipopt, DataFrames, LinearAlgebra, Statistics, CSV


# 1. 系统参数定义
const K = 21                    # 电池总数
const N = 24                    # 预测时域（小时）
const Δt = 1.0                  # 控制步长（小时）
const P_max = 50.0              # 最大充放电功率（kW）
const C_bat = 210.0             # 单块电池容量（kWh）
const SOC_swap_threshold = 0.5  # 换电SOC阈值
const SOC_min = 0.2             # 最小SOC
const SOC_max = 0.8             # 最大SOC

# 电池电化学参数
const C_p_max = 10350.0         # 正极最大浓度 (mol/m³)
const C_n_max = 29480.0         # 负极最大浓度 (mol/m³)
const lp = 6.521e-5             # 正极厚度 (m)
const ln = 2.885e-5             # 负极厚度 (m)
const ep = 1 - 0.52             # 正极体积分数
const en = 1 - 0.619            # 负极体积分数

# 经济参数
const Π = 1000.0                # 容量衰减单位成本 (USD/Ah)
const w1 = 1.0                  # 电池衰减惩罚权重
const w2 = 0.1                  # 利用率均衡惩罚权重

rho = [0.12, 0.11, 0.10, 0.10, 0.11, 0.15, 0.20, 0.18, 0.16, 0.14, 0.13, 0.14,
       0.15, 0.16, 0.17, 0.22, 0.25, 0.23, 0.19, 0.17, 0.15, 0.14, 0.13, 0.12]
#电价数据（USD/kWh）

# 2.Kriging函数模型加载
include("Kriging.jl")  # 假设Kriging.jl包含Kriging模型相关函数
include("SPM_Used_for_Kriging.jl")  # 假设包含SPM参数定义
kriging_model, norm_params = main()  # 从Kriging.jl中获取训练好的模型

# 3. 辅助函数
function soc_to_concentration(soc)
    """将SOC转换为浓度状态"""
    csn_avg = soc * C_n_max
    csp_avg = C_p_max - csn_avg * ln * en / lp / ep
    
    # 确保浓度在合理范围内
    csp_avg = max(0.01 * C_p_max, min(0.99 * C_p_max, csp_avg))
    csn_avg = max(0.01 * C_n_max, min(0.99 * C_n_max, csn_avg))
    
    return csp_avg, csn_avg
end

function get_swap_battery_indices(swap_demand::Vector{Int}, current_swap_index::Int)
    """根据换电需求和当前换电索引确定换电电池序号"""
    swap_indices = Vector{Vector{Int}}()
    current_idx = current_swap_index
    
    for t in 1:length(swap_demand)
        demand = swap_demand[t]
        indices = Int[]
        
        for i in 1:demand
            push!(indices, current_idx)
            current_idx = mod(current_idx, K) + 1  # 循环调度
        end
        
        push!(swap_indices, indices)
    end
    
    return swap_indices, current_idx
end

# 4. 核心优化函数

function solve_bss_optimization(
    kriging_model::KrigingModel,
    electricity_prices::Vector{Float64},
    swap_demand::Vector{Int},
    initial_states::Matrix{Float64},
    current_swap_index::Int
)
    """
    求解BSS优化问题
    initial_states: K*4矩阵,每行是[C_p_avg, C_n_avg, δ_SEI, c_f]
    """
    
    # 创建优化模型
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 5000)
    set_optimizer_attribute(model, "tol", 1e-6)
    set_optimizer_attribute(model, "print_level", 0)
    
    # 获取换电电池序号
    swap_indices, next_swap_index = get_swap_battery_indices(swap_demand, current_swap_index)
    
    
    # 5. 决策变量定义
   
    # 充放电功率 (t=1:N, k=1:K)
    @variable(model, -P_max <= P[1:N, 1:K] <= P_max)
    
    # 电池状态变量 (t=1:N+1, k=1:K)
    @variable(model, C_p_avg[1:N+1, 1:K] >= 0)
    @variable(model, C_n_avg[1:N+1, 1:K] >= 0)
    @variable(model, δ_SEI[1:N+1, 1:K] >= 0)
    @variable(model, c_f[1:N+1, 1:K] >= 0)
    
    # 辅助变量
    @variable(model, P_abs[1:N, 1:K] >= 0)  # 功率绝对值
    
   
    # 6. 初始状态约束
    for k in 1:K
        @constraint(model, C_p_avg[1, k] == initial_states[k, 1])
        @constraint(model, C_n_avg[1, k] == initial_states[k, 2])
        @constraint(model, δ_SEI[1, k] == initial_states[k, 3])
        @constraint(model, c_f[1, k] == initial_states[k, 4])
    end
    

    # 7. 功率绝对值约束

    for t in 1:N, k in 1:K
        @constraint(model, P_abs[t, k] >= P[t, k])
        @constraint(model, P_abs[t, k] >= -P[t, k])
    end
    
    # ===============================
    # 8. 状态转移约束（Kriging模型核心）
    # ===============================
    for t in 1:N, k in 1:K
        # 使用Kriging模型预测状态增量
        # 注意：这里需要将Kriging预测转化为约束
        
        # 正极浓度演化
        @constraint(model, 
            C_p_avg[t+1, k] == C_p_avg[t, k] - 0.0001 * P[t, k] * Δt * (1 + 0.001 * c_f[t, k])
        )
        
        # 负极浓度演化
        @constraint(model,
            C_n_avg[t+1, k] == C_n_avg[t, k] + 0.0001 * P[t, k] * Δt * (1 + 0.001 * c_f[t, k])
        )
        
        # SEI增长
        @constraint(model,
            δ_SEI[t+1, k] == δ_SEI[t, k] + 1e-12 * (1 + 0.1 * (P_abs[t, k]/P_max)^2 + 0.05 * (δ_SEI[t, k]/1e-9)) * Δt
        )
        
        # 容量衰减
        @constraint(model,
            c_f[t+1, k] == c_f[t, k] + 0.1 * (δ_SEI[t+1, k] - δ_SEI[t, k]) * C_n_max / 1e-9
        )
    end
    
    # ===============================
    # 9. 换电SOC约束
    # ===============================
    for t in 1:N
        swap_batteries = swap_indices[t]
        for k in swap_batteries
            # SOC = C_n_avg / C_n_max
            @constraint(model, C_n_avg[t, k] >= SOC_swap_threshold * C_n_max + 0.001 * C_n_max)
        end
    end
    
    # ===============================
    # 10. 目标函数构建
    # ===============================
    
    # 10.1 套利收益
    arbitrage_revenue = @expression(model,
        sum(P[t, k] * electricity_prices[t] * Δt for t in 1:N, k in 1:K)
    )
    
    # 10.2 电池衰减惩罚
    aging_penalty = @expression(model,
        w1 * Π * sum(c_f[t+1, k] - c_f[t, k] for t in 1:N, k in 1:K)
    )
    
    # 10.3 利用率均衡惩罚
    utilization_penalty = @expression(model,
        w2 * sum((c_f[t, k] - sum(c_f[t, j] for j in 1:K)/K)^2 for t in 1:N, k in 1:K)
    )
    
    # 总目标函数
    @objective(model, Max, arbitrage_revenue - aging_penalty - utilization_penalty)
    
    # ===============================
    # 11. 求解优化问题
    # ===============================
    println("开始求解BSS优化问题...")
    optimize!(model)
    
    status = termination_status(model)
    if status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]
        println("优化求解成功!")
        
        # 提取结果
        P_opt = value.(P)
        C_p_avg_opt = value.(C_p_avg)
        C_n_avg_opt = value.(C_n_avg)
        δ_SEI_opt = value.(δ_SEI)
        c_f_opt = value.(c_f)
        
        total_revenue = value(arbitrage_revenue)
        total_aging_cost = value(aging_penalty)
        total_utilization_cost = value(utilization_penalty)
        net_profit = total_revenue - total_aging_cost - total_utilization_cost
        
        # 构建结果字典
        results = Dict(
            :P_opt => P_opt,
            :C_p_avg_opt => C_p_avg_opt,
            :C_n_avg_opt => C_n_avg_opt,
            :δ_SEI_opt => δ_SEI_opt,
            :c_f_opt => c_f_opt,
            :total_revenue => total_revenue,
            :total_aging_cost => total_aging_cost,
            :total_utilization_cost => total_utilization_cost,
            :net_profit => net_profit,
            :next_swap_index => next_swap_index,
            :swap_indices => swap_indices
        )
        
        return results
    else
        error("优化求解失败，状态: $status")
    end
end

# ===============================
# 12. MPC滚动优化主循环
# ===============================
function run_mpc_loop(
    kriging_model::KrigingModel,
    total_hours::Int,
    electricity_prices_full::Vector{Float64},
    swap_demand_full::Vector{Int}
)
    """
    MPC滚动优化主循环
    """
    
    # 初始化电池状态
    initial_states = zeros(K, 4)
    for k in 1:K
        csp_avg, csn_avg = soc_to_concentration(0.5)  # 初始SOC=0.5
        initial_states[k, 1] = csp_avg
        initial_states[k, 2] = csn_avg
        initial_states[k, 3] = 1e-10  # 初始SEI厚度
        initial_states[k, 4] = 0.0    # 初始容量衰减
    end
    
    current_swap_index = 1
    current_states = copy(initial_states)
    
    # 存储结果
    all_results = []
    state_history = []
    push!(state_history, copy(current_states))
    
    for hour in 1:total_hours
        println("MPC循环 - 小时 $hour/$total_hours")
        
        # 获取当前预测时域的数据
        pred_horizon = min(N, total_hours - hour + 1)
        electricity_prices = electricity_prices_full[hour:min(end, hour+pred_horizon-1)]
        swap_demand = swap_demand_full[hour:min(end, hour+pred_horizon-1)]
        
        # 如果预测时域不足，用最后一个值填充
        if length(electricity_prices) < N
            append!(electricity_prices, fill(electricity_prices[end], N - length(electricity_prices)))
            append!(swap_demand, fill(swap_demand[end], N - length(swap_demand)))
        end
        
        # 求解优化问题
        results = solve_bss_optimization(
            kriging_model, electricity_prices, swap_demand, 
            current_states, current_swap_index
        )
        
        push!(all_results, results)
        
     
        # 13. 状态更新（Kriging模型预测）
        # 执行当前时刻决策
        current_power = results[:P_opt][1, :]  # t=1时刻的功率决策
        
        # 使用Kriging模型更新状态
        for k in 1:K
            # 构建输入向量
            x_input = [
                current_states[k, 1],  # C_p_avg
                current_states[k, 2],  # C_n_avg  
                current_states[k, 3],  # δ_SEI
                current_states[k, 4],  # c_f
                current_power[k]       # power
            ]
            
            # Kriging预测状态增量
            Δstate = predict(kriging_model, x_input)
            
            # 更新状态
            current_states[k, 1] += Δstate[1]  # C_p_avg
            current_states[k, 2] += Δstate[2]  # C_n_avg
            current_states[k, 3] += Δstate[3]  # δ_SEI
            current_states[k, 4] += Δstate[4]  # c_f
        end
        
        # 处理换电电池状态重置
        current_swap_batteries = results[:swap_indices][1]  # t=1时刻的换电电池
        for k in current_swap_batteries
            # 换电电池状态重置为新电池状态
            csp_avg_new, csn_avg_new = soc_to_concentration(SOC_swap_threshold)
            current_states[k, 1] = csp_avg_new
            current_states[k, 2] = csn_avg_new
            current_states[k, 3] = 1e-10  # 新电池SEI厚度
            current_states[k, 4] = 0.0    # 新电池容量衰减
        end
        
        # 更新换电索引
        current_swap_index = results[:next_swap_index]
        
        # 保存状态历史
        push!(state_history, copy(current_states))
        
        println("  当前时刻收益: $(round(results[:net_profit], digits=2)) USD")
        println("  换电电池: $current_swap_batteries")
    end
    
    return all_results, state_history
end

# 14. 结果分析和可视化
function analyze_results(all_results, state_history)
    """分析MPC优化结果"""
    
    total_hours = length(all_results)
    
    # 计算总体统计
    total_revenue = sum(r[:total_revenue] for r in all_results)
    total_aging_cost = sum(r[:total_aging_cost] for r in all_results)
    total_utilization_cost = sum(r[:total_utilization_cost] for r in all_results)
    total_profit = total_revenue - total_aging_cost - total_utilization_cost
    
    println("\n" * "="^60)
    println("MPC优化结果总结")
    println("="^60)
    println("总运行时间: $total_hours 小时")
    println("总套利收益: $(round(total_revenue, digits=2)) USD")
    println("总老化成本: $(round(total_aging_cost, digits=2)) USD") 
    println("总均衡惩罚: $(round(total_utilization_cost, digits=2)) USD")
    println("净收益: $(round(total_profit, digits=2)) USD")
    
    # 电池健康状态分析
    final_states = state_history[end]
    avg_capacity_fade = mean(final_states[:, 4])
    max_capacity_fade = maximum(final_states[:, 4])
    avg_sei_thickness = mean(final_states[:, 3]) * 1e9  # 转换为nm
    
    println("\n电池健康状态分析:")
    println("平均容量衰减: $(round(avg_capacity_fade, digits=6)) Ah")
    println("最大容量衰减: $(round(max_capacity_fade, digits=6)) Ah")
    println("平均SEI厚度: $(round(avg_sei_thickness, digits=3)) nm")
    
    return Dict(
        :total_revenue => total_revenue,
        :total_aging_cost => total_aging_cost,
        :total_utilization_cost => total_utilization_cost,
        :total_profit => total_profit,
        :avg_capacity_fade => avg_capacity_fade,
        :max_capacity_fade => max_capacity_fade,
        :avg_sei_thickness => avg_sei_thickness
    )
end


# 15. 主函数

function main()
    println("启动Kriging模型在线嵌入的BSS优化系统")
    
    
    # 运行MPC优化
    println("开始MPC滚动优化...")
    all_results, state_history = run_mpc_loop(
        kriging_model, total_hours, rho, swap_demand
    )
    
    # 分析结果
    summary = analyze_results(all_results, state_history)
    
    println("\n优化完成!")
    return all_results, state_history, summary
end

# 运行主函数
if abspath(PROGRAM_FILE) == @__FILE__
    all_results, state_history, summary = main()
end