using JuMP, Ipopt, DataFrames, LinearAlgebra, Statistics, CSV

# 1. 系统参数定义（移除const关键字）
K = 21                    # 电池总数
N = 1                     # 预测时域（单步优化）
Δt = 1.0                  # 控制步长（小时）
P_max = 50.0              # 最大充放电功率（kW）
C_bat = 210.0             # 单块电池容量（kWh）
SOC_swap_threshold = 0.5  # 换电SOC阈值
SOC_min = 0.2             # 最小SOC
SOC_max = 0.8             # 最大SOC

# 电池电化学参数
C_p_max = 10350.0         # 正极最大浓度 (mol/m³)
C_n_max = 29480.0         # 负极最大浓度 (mol/m³)
lp = 6.521e-5             # 正极厚度 (m)
ln = 2.885e-5             # 负极厚度 (m)
ep = 1 - 0.52             # 正极体积分数
en = 1 - 0.619            # 负极体积分数

# 经济参数
Π = 1000.0                # 容量衰减单位成本 (USD/Ah)
w1 = 1.0                  # 电池衰减惩罚权重
w2 = 0.1                  # 利用率均衡惩罚权重

# 电价数据（USD/kWh）
rho = [0.12, 0.11, 0.10, 0.10, 0.11, 0.15, 0.20, 0.18, 0.16, 0.14, 0.13, 0.14,
       0.15, 0.16, 0.17, 0.22, 0.25, 0.23, 0.19, 0.17, 0.15, 0.14, 0.13, 0.12]

# 2. 辅助函数
function soc_to_concentration(soc)
    """将SOC转换为浓度状态"""
    csn_avg = soc * C_n_max
    csp_avg = C_p_max - csn_avg * ln * en / lp / ep
    
    # 确保浓度在合理范围内
    csp_avg = max(0.01 * C_p_max, min(0.99 * C_p_max, csp_avg))
    csn_avg = max(0.01 * C_n_max, min(0.99 * C_n_max, csn_avg))
    
    return csp_avg, csn_avg
end

function concentration_to_soc(csn_avg)
    """将负极浓度转换为SOC"""
    return min(max(csn_avg / C_n_max, SOC_min), SOC_max)
end

function get_swap_battery_indices(swap_demand::Int, current_swap_index::Int)
    """根据换电需求和当前换电索引确定换电电池序号（单步版本）"""
    swap_indices = Int[]
    current_idx = current_swap_index
    
    for i in 1:swap_demand
        push!(swap_indices, current_idx)
        current_idx = mod(current_idx, K) + 1  # 循环调度
    end
    
    return swap_indices, current_idx
end

# 3. 简化的Kriging预测器
function create_simple_kriging_predictor()
    """创建简化的Kriging预测器"""
    
    function predict_state_increment(state, power)
        # state: [C_p_avg, C_n_avg, δ_SEI, c_f]
        # power: 充放电功率 (kW), 正为放电，负为充电
        
        C_p, C_n, δ_SEI, c_f = state
        
        # 简化的老化模型
        # 注意：这是简化的物理模型，实际应该使用训练好的Kriging模型
        
        # 浓度变化
        ΔC_p = -power * 0.001 * (1 + 0.001 * δ_SEI)  # 充电时正极浓度增加
        ΔC_n = power * 0.001 * (1 + 0.001 * δ_SEI)   # 充电时负极浓度减少
        
        # SEI增长
        Δδ_SEI = abs(power) * 0.0005 * (1 + 0.01 * δ_SEI) * (1 + 0.001 * c_f)
        
        # 容量衰减
        Δc_f = abs(power) * 0.0001 * (1 + 0.01 * δ_SEI) * (1 + 0.001 * c_f)
        
        # 返回小时增量
        return [ΔC_p, ΔC_n, Δδ_SEI, Δc_f]
    end
    
    return predict_state_increment
end

# 4. 单步优化函数
function solve_single_step_optimization(
    kriging_predictor,
    electricity_price::Float64,
    swap_demand::Int,
    initial_states::Matrix{Float64},
    current_swap_index::Int
)
    """
    求解单步BSS优化问题
    
    参数：
    - kriging_predictor: Kriging预测函数
    - electricity_price: 当前电价 (USD/kWh)
    - swap_demand: 当前换电需求
    - initial_states: 初始状态矩阵 (K×4)
    - current_swap_index: 当前换电索引
    """
    println("求解单步BSS优化问题...")
    
    # 获取换电电池序号
    swap_indices, next_swap_index = get_swap_battery_indices(swap_demand, current_swap_index)
    
    # 创建优化模型
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 500)
    set_optimizer_attribute(model, "tol", 1e-4)
    set_optimizer_attribute(model, "print_level", 0)
    
    # 决策变量 - 仅当前时刻的功率
    @variable(model, -P_max <= P[1:K] <= P_max)  # 当前时刻各电池功率
    
    # 中间变量：容量衰减（当前和下一时刻）
    @variable(model, c_f_current[1:K] >= 0)  # 当前时刻容量衰减
    @variable(model, c_f_next[1:K] >= 0)     # 下一时刻容量衰减
    
    # 初始状态约束
    for k in 1:K
        @constraint(model, c_f_current[k] == initial_states[k, 4])
    end
    
    # 功率平衡约束（总功率不能超过系统限制）
    @constraint(model, -P_max*K <= sum(P[k] for k=1:K) <= P_max*K)
    
    # 更换电池的SOC约束
    for k in swap_indices
        current_soc = concentration_to_soc(initial_states[k, 2])
        if current_soc < SOC_swap_threshold
            println("  警告：电池 $k 的SOC ($(round(current_soc*100,1))%) 低于换电阈值 ($(SOC_swap_threshold*100)%)")
        end
    end
    
    # 目标函数组件
    # 1. 套利收益（当前时刻）
    arbitrage_revenue = @expression(model,
        sum(P[k] * electricity_price * Δt for k in 1:K)
    )
    
    # 2. 电池退化惩罚（当前时刻的容量衰减增量）
    degradation_penalty = @expression(model,
        Π * sum(c_f_next[k] - c_f_current[k] for k in 1:K)
    )
    
    # 3. 使用均衡惩罚（基于当前容量衰减）
    avg_cf = @expression(model, sum(c_f_current[k] for k in 1:K) / K)
    utilization_penalty = @expression(model,
        sum((c_f_current[k] - avg_cf)^2 for k in 1:K)
    )
    
    # 4. 容量衰减动态（简化线性模型）
    for k in 1:K
        @constraint(model, 
            c_f_next[k] == c_f_current[k] + 0.0001 * abs(P[k]) * Δt
        )
    end
    
    # 总目标函数：最大化净收益
    @objective(model, Max, arbitrage_revenue - w1*degradation_penalty - w2*utilization_penalty)
    
    # 求解优化问题
    println("  使用Ipopt求解器进行优化...")
    optimize!(model)
    
    status = termination_status(model)
    if status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]
        println("  优化求解成功!")
        
        # 提取结果
        P_opt = value.(P)
        c_f_current_opt = value.(c_f_current)
        c_f_next_opt = value.(c_f_next)
        
        # 使用Kriging模型计算状态更新
        C_p_next = zeros(K)
        C_n_next = zeros(K)
        δ_SEI_next = zeros(K)
        
        for k in 1:K
            if k in swap_indices
                # 更换电池：状态重置
                C_p_next[k], C_n_next[k] = soc_to_concentration(0.8)
                δ_SEI_next[k] = 1e-9
            else
                # 非更换电池：使用Kriging模型更新状态
                power = P_opt[k]
                current_state = [initial_states[k, 1], initial_states[k, 2], 
                                 initial_states[k, 3], c_f_current_opt[k]]
                delta_state = kriging_predictor(current_state, power)
                
                C_p_next[k] = initial_states[k, 1] + delta_state[1] * Δt
                C_n_next[k] = initial_states[k, 2] + delta_state[2] * Δt
                δ_SEI_next[k] = initial_states[k, 3] + delta_state[3] * Δt
            end
        end
        
        total_revenue = value(arbitrage_revenue)
        total_degradation = value(degradation_penalty)
        total_utilization = value(utilization_penalty)
        
        # 构建结果字典
        results = Dict(
            :P_opt => P_opt,
            :C_p_next => C_p_next,
            :C_n_next => C_n_next,
            :δ_SEI_next => δ_SEI_next,
            :c_f_current => c_f_current_opt,
            :c_f_next => c_f_next_opt,
            :total_revenue => total_revenue,
            :degradation_cost => total_degradation,
            :utilization_penalty => total_utilization,
            :objective_value => objective_value(model),
            :next_swap_index => next_swap_index,
            :swap_indices => swap_indices
        )
        
        return results
    else
        println("  优化求解失败，状态: $status")
        # 返回保守的控制策略（零功率）
        return Dict(
            :P_opt => zeros(K),
            :next_swap_index => current_swap_index,
            :swap_indices => swap_indices,
            :total_revenue => 0.0,
            :degradation_cost => 0.0,
            :utilization_penalty => 0.0,
            :objective_value => 0.0
        )
    end
end

# 5. 模拟运行函数（单步执行）
function simulate_single_step()
    """
    模拟单步优化执行
    """
    println("="^60)
    println("BSS单步优化模拟")
    println("="^60)
    
    # 创建Kriging预测器
    println("1. 创建Kriging预测器...")
    kriging_predictor = create_simple_kriging_predictor()
    
    # 测试Kriging预测器
    println("\n2. 测试Kriging预测器...")
    test_state = [6520.5, 11202.4, 15.2, 21.5]
    test_power = -23.1
    pred = kriging_predictor(test_state, test_power)
    println("  测试预测结果:")
    println("    ΔC_p: $(round(pred[1],6)) mol/m³")
    println("    ΔC_n: $(round(pred[2],6)) mol/m³")
    println("    Δδ_SEI: $(round(pred[3],6)) nm")
    println("    Δc_f: $(round(pred[4],6)) Ah")
    
    # 初始化电池状态
    println("\n3. 初始化电池状态...")
    initial_states = zeros(K, 4)
    for k in 1:K
        csp_avg, csn_avg = soc_to_concentration(0.5)
        initial_states[k, 1] = csp_avg
        initial_states[k, 2] = csn_avg
        initial_states[k, 3] = 1e-9
        initial_states[k, 4] = 0.0
        if k <= 3  # 显示前3个电池的状态
            println("  电池 $k: C_p=$(round(csp_avg,1)), C_n=$(round(csn_avg,1)), SOC=$(round(concentration_to_soc(csn_avg)*100,1))%")
        end
    end
    
    # 设置当前时刻参数
    current_time = 1  # 上午8点（电价高峰）
    current_price = rho[current_time]
    current_swap_demand = 2  # 需要更换2块电池
    current_swap_index = 1
    
    println("\n4. 设置优化参数:")
    println("   当前时刻: $current_time (电价: \$$current_price/kWh)")
    println("   换电需求: $current_swap_demand 块电池")
    println("   当前换电索引: $current_swap_index")
    
    # 执行单步优化
    println("\n5. 执行单步优化...")
    results = solve_single_step_optimization(
        kriging_predictor, current_price, current_swap_demand,
        initial_states, current_swap_index
    )
    
    # 显示优化结果
    println("\n6. 优化结果:")
    println("   套利收益: \$$(round(results[:total_revenue], 2))")
    println("   退化成本: \$$(round(results[:degradation_cost], 2))")
    println("   均衡惩罚: $(round(results[:utilization_penalty], 4))")
    println("   目标函数值: \$$(round(results[:objective_value], 2))")
    
    # 分析功率分配
    println("\n7. 功率分配分析:")
    P_opt = results[:P_opt]
    charging_count = sum(P_opt .< 0)
    discharging_count = sum(P_opt .> 0)
    idle_count = sum(P_opt .== 0)
    total_power = sum(P_opt)
    
    println("   充电电池数: $charging_count")
    println("   放电电池数: $discharging_count")
    println("   空闲电池数: $idle_count")
    println("   总功率: $(round(total_power, 2)) kW")
    
    # 显示前5个电池的优化结果
    println("\n8. 前5个电池优化详情:")
    for k in 1:min(5, K)
        power = P_opt[k]
        action = power < 0 ? "充电" : (power > 0 ? "放电" : "空闲")
        println("   电池 $k: $(round(power, 2)) kW ($action)")
    end
    
    # 显示更换电池信息
    println("\n9. 更换电池信息:")
    swap_indices = results[:swap_indices]
    if !isempty(swap_indices)
        println("   需要更换的电池: $(join(swap_indices, ", "))")
        println("   下一换电索引: $(results[:next_swap_index])")
    else
        println("   本时刻无电池更换")
    end
    
    # 显示状态更新结果
    println("\n10. 状态更新示例（电池1）:")
    println("    初始状态: C_p=$(round(initial_states[1,1],1)), C_n=$(round(initial_states[1,2],1)), " *
            "δ_SEI=$(round(initial_states[1,3],6)), c_f=$(round(initial_states[1,4],6))")
    println("    更新后状态: C_p=$(round(results[:C_p_next][1],1)), C_n=$(round(results[:C_n_next][1],1)), " *
            "δ_SEI=$(round(results[:δ_SEI_next][1],6)), c_f=$(round(results[:c_f_next][1],6))")
    
    # 保存结果
    println("\n11. 保存优化结果...")
    results_df = DataFrame(
        battery = 1:K,
        power_kW = P_opt,
        revenue_USD = fill(results[:total_revenue], K),
        degradation_cost_USD = fill(results[:degradation_cost], K),
        utilization_penalty = fill(results[:utilization_penalty], K)
    )
    
    CSV.write("single_step_optimization_results.csv", results_df)
    println("    结果已保存到 single_step_optimization_results.csv")
    
    return results
end

# 6. 批量运行测试
function run_batch_tests()
    """
    运行多个单步优化测试
    """
    println("\n" * "="^60)
    println("批量优化测试")
    println("="^60)
    
    # 创建Kriging预测器
    kriging_predictor = create_simple_kriging_predictor()
    
    # 初始化电池状态
    initial_states = zeros(K, 4)
    for k in 1:K
        csp_avg, csn_avg = soc_to_concentration(0.5)
        initial_states[k, 1] = csp_avg
        initial_states[k, 2] = csn_avg
        initial_states[k, 3] = 1e-9
        initial_states[k, 4] = 0.0
    end
    
    current_swap_index = 1
    
    # 测试不同时间的优化
    test_cases = [
        (1, 0.12, 2, "上午8点 (低电价, 高峰需求)"),
        (8, 0.20, 1, "下午4点 (高电价, 中等需求)"),
        (15, 0.17, 0, "晚上11点 (中等电价, 低需求)"),
        (20, 0.17, 3, "凌晨4点 (中等电价, 高峰需求)")
    ]
    
    all_results = []
    
    for (time_idx, price, swap_demand, desc) in test_cases
        println("\n测试: $desc")
        println("  时间索引: $time_idx, 电价: \$$price/kWh, 换电需求: $swap_demand")
        
        results = solve_single_step_optimization(
            kriging_predictor, price, swap_demand,
            initial_states, current_swap_index
        )
        
        # 更新换电索引和状态（模拟）
        current_swap_index = results[:next_swap_index]
        
        # 更新初始状态用于下一个测试
        for k in 1:K
            initial_states[k, 1] = results[:C_p_next][k]
            initial_states[k, 2] = results[:C_n_next][k]
            initial_states[k, 3] = results[:δ_SEI_next][k]
            initial_states[k, 4] = results[:c_f_next][k]
        end
        
        push!(all_results, (desc=desc, results=results))
        
        println("  优化结果: 收益=\$$(round(results[:total_revenue],2)), " *
                "成本=\$$(round(results[:degradation_cost],2)), " *
                "净收益=\$$(round(results[:objective_value],2))")
    end
    
    # 汇总分析
    println("\n" * "-"^60)
    println("批量测试汇总:")
    total_revenue = sum(r.results[:total_revenue] for r in all_results)
    total_cost = sum(r.results[:degradation_cost] for r in all_results)
    total_net = sum(r.results[:objective_value] for r in all_results)
    
    println("  总收益: \$$(round(total_revenue, 2))")
    println("  总退化成本: \$$(round(total_cost, 2))")
    println("  总净收益: \$$(round(total_net, 2))")
    
    return all_results
end

# 7. 主函数
function main()
    println("基于Kriging代理模型的BSS单步优化系统")
    println("="^60)
    
    # 运行单步优化模拟
    single_results = simulate_single_step()
    
    println("\n" * "="^60)
    println("是否运行批量测试? (输入 'y' 继续, 其他键退出)")
    user_input = readline()
    
    if lowercase(strip(user_input)) == "y"
        batch_results = run_batch_tests()
        println("\n批量测试完成!")
    else
        println("\n跳过批量测试")
    end
    
    println("\n" * "="^60)
    println("程序执行完成!")
    println("="^60)
    
    return single_results
end

# 运行主函数
if abspath(PROGRAM_FILE) == @__FILE__
    println("开始执行BSS单步优化系统...")
    
    # 检查必要的包
    try
        using JuMP, Ipopt, DataFrames, CSV
        println("必要的包已加载")
    catch e
        println("错误: 缺少必要的包")
        println("请使用以下命令安装:")
        println("using Pkg")
        println("Pkg.add([\"JuMP\", \"Ipopt\", \"DataFrames\", \"CSV\"])")
        return
    end
    
    # 运行主函数
    try
        main()
    catch e
        println("程序执行出错: $e")
        showerror(stdout, e)
        println("\n请检查输入参数和优化模型。")
    end
end