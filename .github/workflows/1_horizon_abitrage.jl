using JuMP, Ipopt, DataFrames, LinearAlgebra, Statistics, CSV

# 包含Kriging模型定义和训练
include("D:\\vscode codes\\demo-script\\.github\\workflows\\Kriging_module.jl")

# 系统参数
const K = 21                    # 电池总数
const N = 24                    # 预测时域（小时）
const Δt = 1.0                  # 控制步长（小时）
const P_max = 50.0              # 最大充放电功率（kW）
const SOC_swap_threshold = 0.5  # 换电SOC阈值
const SOC_min = 0.2             # 最小SOC
const SOC_max = 0.8             # 最大SOC

# 电池电化学参数
const C_n_max = 29480.0         # 负极最大浓度 (mol/m³)

# 经济参数
const Π = 1000.0                # 容量衰减单位成本 (USD/Ah)
const w1 = 1.0                  # 电池衰减惩罚权重
const w2 = 0.1                  # 利用率均衡惩罚权重

# 电价数据（USD/kWh）
rho = [0.12, 0.11, 0.10, 0.10, 0.11, 0.15, 0.20, 0.18, 0.16, 0.14, 0.13, 0.14,
             0.15, 0.16, 0.17, 0.22, 0.25, 0.23, 0.19, 0.17, 0.15, 0.14, 0.13, 0.12]

# 辅助函数：SOC计算
function concentration_to_soc(csn_avg)
    """将负极浓度转换为SOC"""
    return min(max(csn_avg / C_n_max, SOC_min), SOC_max)
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

# 主优化函数
function solve_bss_optimization(
    kriging_predictor,
    electricity_prices::Vector{Float64},
    swap_demand::Vector{Int},
    initial_states::Matrix{Float64},
    current_swap_index::Int
)
    """
    求解BSS优化问题 - 简化版
    使用外部Kriging预测,避免在JuMP中集成复杂非线性函数
    """
    println("开始求解BSS优化问题...")
    
    # 获取换电电池序号
    swap_indices, next_swap_index = get_swap_battery_indices(swap_demand, current_swap_index)
    
    # 创建优化模型
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 1000)
    set_optimizer_attribute(model, "tol", 1e-6)
    set_optimizer_attribute(model, "print_level", 0)
    
    # 决策变量
    @variable(model, -P_max <= P[1:N, 1:K] <= P_max)  # 功率
    
    # 中间变量：状态变量（为了简化，不在优化变量中定义完整状态转移）
    # 我们使用Kriging预测作为约束的参考，但不在优化中直接求解微分方程
    @variable(model, c_f[1:N+1, 1:K] >= 0)  # 容量衰减
    
    # 初始状态约束
    for k in 1:K
        @constraint(model, c_f[1, k] == initial_states[k, 4])
    end
    
    # 简化约束：总功率平衡
    @constraint(model, [t=1:N], -P_max*K <= sum(P[t, k] for k=1:K) <= P_max*K)
    
    # 更换电池的SOC约束（简化版本）
    for t in 1:N
        for k in swap_indices[t]
            # 使用当前状态计算SOC
            current_soc = concentration_to_soc(initial_states[k, 2])
            # 确保换电时SOC足够
            @constraint(model, current_soc >= SOC_swap_threshold)
        end
    end
    
    # 目标函数组件
    # 1. 套利收益
    arbitrage_revenue = @expression(model,
        sum(P[t, k] * electricity_prices[t] * Δt for t in 1:N, k in 1:K)
    )
    
    # 2. 电池退化惩罚（基于功率的简化模型）
    # 假设退化与功率的绝对值成比例
    degradation_penalty = @expression(model,
        Π * sum(abs(P[t, k]) * 0.001 * Δt for t in 1:N, k in 1:K)  # 简化模型
    )
    
    # 3. 使用均衡惩罚（基于容量的变化）
    utilization_penalty = @expression(model,
        w2 * sum((c_f[t, k] - sum(c_f[t, j] for j in 1:K)/K)^2 
                for t in 2:N+1, k in 1:K)
    )
    
    # 4. 容量衰减动态（简化线性模型）
    for t in 1:N, k in 1:K
        # 简化：容量衰减与功率绝对值成正比
        @constraint(model, 
            c_f[t+1, k] == c_f[t, k] + 0.001 * abs(P[t, k]) * Δt
        )
    end
    
    # 总目标函数：最大化净收益
    @objective(model, Max, arbitrage_revenue - degradation_penalty - utilization_penalty)
    
    # 求解优化问题
    println("求解NLP优化问题...")
    optimize!(model)
    
    status = termination_status(model)
    if status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]
        println("优化求解成功!")
        
        # 提取结果
        P_opt = value.(P)
        c_f_opt = value.(c_f)
        
        # 使用Kriging模型进行后验状态更新
        # 这里我们使用Kriging预测器来更新状态，而不是在优化中
        C_p_opt = zeros(N+1, K)
        C_n_opt = zeros(N+1, K)
        δ_SEI_opt = zeros(N+1, K)
        
        # 设置初始状态
        for k in 1:K
            C_p_opt[1, k] = initial_states[k, 1]
            C_n_opt[1, k] = initial_states[k, 2]
            δ_SEI_opt[1, k] = initial_states[k, 3]
        end
        
        # 使用Kriging模型进行状态更新
        for t in 1:N, k in 1:K
            if !(k in swap_indices[t])
                # 非更换电池：使用Kriging更新
                power = P_opt[t, k]
                current_state = [C_p_opt[t, k], C_n_opt[t, k], δ_SEI_opt[t, k], c_f_opt[t, k]]
                delta_state = kriging_predictor(current_state, power)
                
                C_p_opt[t+1, k] = C_p_opt[t, k] + delta_state[1] * Δt
                C_n_opt[t+1, k] = C_n_opt[t, k] + delta_state[2] * Δt
                δ_SEI_opt[t+1, k] = δ_SEI_opt[t, k] + delta_state[3] * Δt
            else
                # 更换电池：状态重置
                C_p_opt[t+1, k] = initial_states[k, 1]
                C_n_opt[t+1, k] = initial_states[k, 2]
                δ_SEI_opt[t+1, k] = initial_states[k, 3]
            end
        end
        
        total_revenue = value(arbitrage_revenue)
        total_degradation = value(degradation_penalty)
        total_utilization = value(utilization_penalty)
        
        # 构建结果字典
        results = Dict(
            :P_opt => P_opt,
            :C_p_avg => C_p_opt,
            :C_n_avg => C_n_opt,
            :δ_SEI => δ_SEI_opt,
            :c_f => c_f_opt,
            :total_revenue => total_revenue,
            :degradation_cost => total_degradation,
            :utilization_penalty => total_utilization,
            :objective_value => objective_value(model),
            :next_swap_index => next_swap_index,
            :swap_indices => swap_indices
        )
        
        return results
    else
        println("优化求解失败，状态: $status")
        # 返回保守的控制策略
        return Dict(
            :P_opt => zeros(N, K),
            :next_swap_index => current_swap_index,
            :swap_indices => swap_indices,
            :total_revenue => 0.0,
            :degradation_cost => 0.0,
            :utilization_penalty => 0.0,
            :objective_value => 0.0
        )
    end
end

# MPC滚动优化循环
function run_mpc_loop(
    kriging_predictor,
    total_hours::Int,
    electricity_prices_full::Vector{Float64},
    swap_demand_full::Vector{Int}
)
    """
    简化的MPC滚动优化循环
    """
    println("启动MPC滚动优化...")
    
    # 初始化电池状态
    initial_states = zeros(K, 4)
    for k in 1:K
        # 简单初始化
        initial_states[k, 1] = 8000.0  # C_p_avg
        initial_states[k, 2] = 15000.0 # C_n_avg
        initial_states[k, 3] = 10.0    # δ_SEI (nm)
        initial_states[k, 4] = 0.0     # c_f
    end
    
    current_swap_index = 1
    current_states = copy(initial_states)
    
    # 存储结果
    all_results = []
    state_history = []
    push!(state_history, copy(current_states))
    
    hour = 1
    while hour <= total_hours
        println("\nMPC循环 - 小时 $hour/$total_hours")
        
        # 获取当前预测时域的数据
        t_start = hour
        t_end = min(hour + N - 1, total_hours)
        n_steps = t_end - t_start + 1
        
        if n_steps < 1
            break
        end
        
        electricity_prices = electricity_prices_full[t_start:t_end]
        swap_demand = swap_demand_full[t_start:t_end]
        
        # 如果序列长度不足N，用最后一个值填充
        if length(electricity_prices) < N
            electricity_prices = vcat(electricity_prices, 
                                     fill(electricity_prices[end], N - length(electricity_prices)))
        end
        if length(swap_demand) < N
            swap_demand = vcat(swap_demand, 
                              fill(swap_demand[end], N - length(swap_demand)))
        end
        
        # 求解优化问题
        results = solve_bss_optimization(
            kriging_predictor, electricity_prices, swap_demand, 
            current_states, current_swap_index
        )
        
        push!(all_results, results)
        
        # 执行第一步决策
        P_opt = results[:P_opt]
        
        # 状态更新
        for k in 1:K
            # 检查电池k是否在第一个时间步被更换
            if k in results[:swap_indices][1]
                # 被更换电池状态重置
                current_states[k, :] = initial_states[k, :]
            else
                # 使用Kriging模型更新状态
                power = P_opt[1, k]
                delta_state = kriging_predictor(current_states[k, :], power)
                
                current_states[k, 1] += delta_state[1] * Δt
                current_states[k, 2] += delta_state[2] * Δt
                current_states[k, 3] += delta_state[3] * Δt
                current_states[k, 4] += delta_state[4] * Δt
            end
        end
        
        # 更新换电索引
        current_swap_index = results[:next_swap_index]
        
        # 保存状态历史
        push!(state_history, copy(current_states))
        
        # 输出当前时刻结果
        println("  当前时刻收益: $(round(results[:total_revenue], digits=2)) USD")
        
        hour += 1
    end
    
    return all_results, state_history
end

# 分析结果
function analyze_results(all_results, state_history)
    """分析优化结果"""
    
    println("\n" * "="^60)
    println("优化结果分析")
    println("="^60)
    
    # 计算总收益和总成本
    total_revenue = sum(r[:total_revenue] for r in all_results)
    total_degradation = sum(r[:degradation_cost] for r in all_results)
    total_utilization = sum(r[:utilization_penalty] for r in all_results)
    net_profit = total_revenue - total_degradation - total_utilization
    
    println("总套利收益: $(round(total_revenue, digits=2)) USD")
    println("总电池退化成本: $(round(total_degradation, digits=2)) USD")
    println("总使用均衡惩罚: $(round(total_utilization, digits=4))")
    println("净收益: $(round(net_profit, digits=2)) USD")
    
    return Dict(
        :total_revenue => total_revenue,
        :total_degradation => total_degradation,
        :total_utilization => total_utilization,
        :net_profit => net_profit
    )
end

# 主函数
function main()
    println("启动基于Kriging代理模型的BSS优化系统")
    println("="^60)
    
    # 1. 训练Kriging模型
    println("\n1. 训练Kriging代理模型...")
    model, norm_params = main()  # 调用Kriging.jl中的main函数
    
    # 2. 创建Kriging预测器
    println("\n2. 创建Kriging预测器...")
    kriging_predictor = use_kriging_model(model, norm_params)
    
    # 3. 测试Kriging预测器
    println("\n3. 测试Kriging预测器...")
    test_state = [6520.5, 11202.4, 15.2, 21.5]
    test_power = -23.1
    pred = kriging_predictor(test_state, test_power)
    println("  测试预测结果:")
    println("    ΔC_p: $(round(pred[1],6))")
    println("    ΔC_n: $(round(pred[2],6))")
    println("    Δδ_SEI: $(round(pred[3],6)) pm")
    println("    Δc_f: $(round(pred[4],6)) μAh")
    
    # 4. 运行MPC滚动优化
    println("\n4. 启动MPC滚动优化...")
    
    # 创建测试数据
    total_hours = 24  # 测试24小时
    swap_demand = [1, 0, 2, 0, 1, 0, 2, 1, 0, 1, 0, 2, 
                   1, 0, 1, 0, 2, 1, 0, 1, 0, 2, 1, 0]
    
    # 运行MPC循环
    all_results, state_history = run_mpc_loop(
        kriging_predictor, total_hours, rho, swap_demand
    )
    
    # 5. 分析结果
    println("\n5. 分析优化结果...")
    summary = analyze_results(all_results, state_history)
    
    println("\n" * "="^60)
    println("优化完成!")
    println("="^60)
    
    # 保存结果
    println("\n保存优化结果...")
    results_df = DataFrame(
        hour = 1:length(all_results),
        revenue = [r[:total_revenue] for r in all_results],
        degradation = [r[:degradation_cost] for r in all_results],
        utilization = [r[:utilization_penalty] for r in all_results],
        objective_value = [r[:objective_value] for r in all_results]
    )
    
    CSV.write("bss_optimization_results.csv", results_df)
    println("结果已保存到 bss_optimization_results.csv")
    
    return all_results, state_history, summary
end

# 运行主函数
if abspath(PROGRAM_FILE) == @__FILE__
    println("基于Kriging代理模型的BSS优化系统")
    println("使用Ipopt求解器")
    println("="^60)
    
    try
        all_results, state_history, summary = main()
        
        # 输出最终摘要
        println("\n最终优化摘要:")
        println("总运行小时数: $(length(all_results))")
        println("净收益: $(round(summary[:net_profit], 2)) USD")
        
    catch e
        println("程序执行出错: $e")
        println("错误类型: $(typeof(e))")
        println("切换到简化模式...")
        
        # 简化模式：使用测试数据
        println("\n使用测试数据运行简化版本...")
        
        # 创建简化的Kriging预测器
        function simple_kriging_predictor(state, power)
            # 简化预测模型
            return [power * 0.001, -power * 0.001, abs(power) * 0.01, abs(power) * 0.0001]
        end
        
        # 运行简化MPC
        total_hours = 12
        swap_demand = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        
        simple_results, simple_history = run_mpc_loop(
            simple_kriging_predictor, total_hours, rho[1:12], swap_demand
        )
        
        println("\n简化版本运行完成!")
        total_rev = sum(r[:total_revenue] for r in simple_results)
        println("总收益: $(round(total_rev, 2)) USD")
    end
end