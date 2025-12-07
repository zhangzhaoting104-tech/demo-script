module BSSMPCFramework

using JuMP, Ipopt, LinearAlgebra, Statistics, CSV, DataFrames

# 导入Kriging模块
include("Kriging_module.jl")
using .KrigingModelModule

# ===================== 数据结构定义 =====================

struct BatteryState
    C_p_avg::Float64      # 正极平均浓度
    C_n_avg::Float64      # 负极平均浓度
    δ_SEI::Float64        # SEI层厚度
    c_f::Float64          # 容量衰减因子
end

# ===================== 核心MPC函数 =====================

function create_test_scenario(;
    K::Int=5,             # 电池数量
    N::Int=24,            # 预测时域长度
    total_hours::Int=72,  # 总仿真时长
    data_path::String="spm_training_data.csv",
    w1::Float64=0.1,      # 退化惩罚权重
    w2::Float64=0.01,     # 均衡惩罚权重
    P1_penalty::Float64=100.0
)
    """
    创建MPC测试场景
    """
    println("="^60)
    println("创建BSS-MPC测试场景")
    println("="^60)
    
    # 1. 训练Kriging模型
    println("初始化Kriging模型...")
    _, kriging_predictor = create_and_train_kriging_model(data_path, verbose=false)
    
    # 2. 初始化电池状态 - 使用具体类型Vector{BatteryState}
    batteries = Vector{BatteryState}(undef, K)
    for k in 1:K
        batteries[k] = BatteryState(
            6500.0 + 500*randn(),    # C_p_avg
            11000.0 + 1000*randn(),  # C_n_avg
            10.0 + 5*rand(),         # δ_SEI
            20.0 + 5*rand()          # c_f
        )
    end
    
    # 3. 创建电价序列
    println("生成电价序列...")
    electricity_prices = Vector{Float64}(undef, total_hours + N)
    for hour in 1:(total_hours+N)
        # 峰谷电价模式
        hour_of_day = hour % 24
        if 9 <= hour_of_day <= 17  # 白天高峰时段
            price = 0.15 + 0.05*randn()
        else  # 夜间低谷时段
            price = 0.08 + 0.02*randn()
        end
        electricity_prices[hour] = price
    end
    
    # 4. 创建更换需求序列
    println("生成电池更换需求...")
    swap_demand = zeros(Int, total_hours+N)
    for hour in 1:24:length(swap_demand)
        if hour + 6 < length(swap_demand)
            swap_demand[hour+6] = min(2, K)
        end
    end
    
    # 5. 返回控制器数据结构
    controller = Dict{String, Any}(
        "batteries" => batteries,
        "K" => K,
        "N" => N,
        "total_hours" => total_hours,
        "current_time" => 1,
        "kriging_predictor" => kriging_predictor,
        "electricity_prices" => electricity_prices,
        "swap_demand" => swap_demand,
        "swap_order" => collect(1:K),
        "w1" => w1,
        "w2" => w2,
        "P1_penalty" => P1_penalty,
        "power_history" => Vector{Vector{Float64}}(),
        "state_history" => Dict{Int, Vector{BatteryState}}(),
        "objective_history" => Vector{Float64}()
    )
    
    println("场景创建完成!")
    println("电池数量: $K")
    println("预测时域: $N 小时")
    println("总仿真时长: $total_hours 小时")
    
    return controller
end

function run_mpc_step(controller::Dict{String, Any}, current_states::Vector{BatteryState})
    """
    执行单步MPC优化
    """
    K = controller["K"]::Int
    N = controller["N"]::Int
    current_time = controller["current_time"]::Int
    
    # 检查是否还有足够的数据
    if current_time + N - 1 > length(controller["electricity_prices"])
        println("数据不足，结束优化")
        return false, zeros(K), 0.0
    end
    
    # 创建优化模型
    model = Model(Ipopt.Optimizer)
    set_silent(model)  # 减少输出
    
    # 决策变量：未来N个时刻的功率
    @variable(model, P[1:N, 1:K])
    
    # 物理约束：功率上下限
    for t in 1:N, k in 1:K
        @constraint(model, P[t, k] >= -50)  # 最小功率（充电）
        @constraint(model, P[t, k] <= 50)   # 最大功率（放电）
    end
    
    # 目标函数：最大化套利收益
    revenue = @expression(model, 
        sum(P[1, k] * (controller["electricity_prices"]::Vector{Float64})[current_time] 
            for k in 1:K)
    )
    
    @objective(model, Max, revenue)
    
    # 求解优化问题
    optimize!(model)
    
    # 检查求解状态
    status = termination_status(model)
    if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
        # 提取最优功率（只取第一时刻的决策）
        P_opt = zeros(K)
        for k in 1:K
            P_opt[k] = value(P[1, k])
        end
        
        return true, P_opt, objective_value(model)
    else
        println("优化失败，状态: $status")
        return false, zeros(K), 0.0
    end
end

function update_with_kriging(state::BatteryState, power::Float64, predictor)
    """
    使用Kriging模型更新单个电池状态
    """
    state_vec = [state.C_p_avg, state.C_n_avg, state.δ_SEI, state.c_f]
    delta = predictor(state_vec, power)
    
    return BatteryState(
        state.C_p_avg + delta[1],
        state.C_n_avg + delta[2],
        state.δ_SEI + delta[3],
        state.c_f + delta[4]
    )
end

function run_mpc_rolling(controller::Dict{String, Any})
    """
    执行MPC滚动时域优化
    """
    println("\n" * "="^60)
    println("开始MPC滚动优化")
    println("="^60)
    
    K = controller["K"]::Int
    total_hours = controller["total_hours"]::Int
    predictor = controller["kriging_predictor"]
    
    # 初始化当前状态
    current_states = controller["batteries"]::Vector{BatteryState}
    
    for hour in 1:total_hours
        print("时间步 $hour/$total_hours: ")
        
        # 执行MPC优化
        success, P_opt, obj_val = run_mpc_step(controller, current_states)
        
        if success
            # 使用Kriging模型更新状态
            new_states = Vector{BatteryState}(undef, K)
            for k in 1:K
                new_states[k] = update_with_kriging(
                    current_states[k], 
                    P_opt[k], 
                    predictor
                )
            end
            
            # 处理电池更换
            if (controller["swap_demand"]::Vector{Int})[hour] > 0
                swap_count = min((controller["swap_demand"]::Vector{Int})[hour], K)
                for k in 1:swap_count
                    new_states[k] = BatteryState(6500.0, 11000.0, 10.0, 20.0)
                    println("电池 $k 已更换")
                end
            end
            
            # 更新当前状态
            current_states = new_states
            
            # 保存结果
            push!(controller["power_history"]::Vector{Vector{Float64}}, P_opt)
            push!(controller["objective_history"]::Vector{Float64}, obj_val)
            
            # 保存状态历史
            for k in 1:K
                state_history = controller["state_history"]::Dict{Int, Vector{BatteryState}}
                if !haskey(state_history, k)
                    state_history[k] = BatteryState[]
                end
                push!(state_history[k], current_states[k])
            end
            
            # 修复round函数调用
            println("成功 | 收益: \$$(round(obj_val, digits=2)) | 平均功率: $(round(mean(abs.(P_opt)), digits=2)) kW")
        else
            # 优化失败，使用备用策略（零功率）
            push!(controller["power_history"]::Vector{Vector{Float64}}, zeros(K))
            push!(controller["objective_history"]::Vector{Float64}, 0.0)
            println("失败 | 使用备用策略")
        end
        
        # 更新时间
        controller["current_time"] += 1
    end
    
    # 更新最终状态
    controller["batteries"] = current_states
    
    println("\nMPC滚动优化完成!")
    return controller
end

function analyze_mpc_results(controller::Dict{String, Any})
    """
    分析MPC结果
    """
    println("\n" * "="^60)
    println("MPC结果分析")
    println("="^60)
    
    # 计算总收益
    total_revenue = sum(controller["objective_history"]::Vector{Float64})
    println("总套利收益: \$$(round(total_revenue, digits=2))")
    
    # 计算成功率
    successful_steps = sum(x > 0 for x in controller["objective_history"]::Vector{Float64})
    success_rate = successful_steps / length(controller["objective_history"]) * 100
    println("优化成功率: $(round(success_rate, digits=1))%")
    
    # 分析功率使用情况
    power_history = controller["power_history"]::Vector{Vector{Float64}}
    if !isempty(power_history)
        all_powers = vcat([reshape(p, 1, :) for p in power_history]...)
        avg_charge = mean(all_powers[all_powers .< 0])
        avg_discharge = mean(all_powers[all_powers .> 0])
        println("平均充电功率: $(round(avg_charge, digits=2)) kW")
        println("平均放电功率: $(round(avg_discharge, digits=2)) kW")
    end
    
    # 显示电池最终状态
    println("\n电池最终状态:")
    batteries = controller["batteries"]::Vector{BatteryState}
    for (k, battery) in enumerate(batteries)
        println("电池 $k: C_p=$(round(battery.C_p_avg, digits=1)), " *
                "C_n=$(round(battery.C_n_avg, digits=1)), " *
                "δ_SEI=$(round(battery.δ_SEI, digits=2))nm, " *
                "c_f=$(round(battery.c_f, digits=2))Ah")
    end
    
    # 返回分析结果
    return Dict(
        "total_revenue" => total_revenue,
        "success_rate" => success_rate,
        "battery_states" => batteries
    )
end

function visualize_power_decisions(controller::Dict{String, Any}, max_steps::Int=10)
    """
    可视化功率决策
    """
    println("\n前$max_steps 个时间步的功率决策:")
    println("-"^50)
    
    power_history = controller["power_history"]::Vector{Vector{Float64}}
    objective_history = controller["objective_history"]::Vector{Float64}
    
    for t in 1:min(max_steps, length(power_history))
        powers = power_history[t]
        charge_count = sum(powers .< 0)
        discharge_count = sum(powers .> 0)
        idle_count = sum(powers .== 0)
        
        println("时间步 $t:")
        println("  充电电池: $charge_count 个, 放电电池: $discharge_count 个, 空闲电池: $idle_count 个")
        println("  功率分布: $(round.(powers, digits=2))")
        println("  收益: \$$(round(objective_history[t], digits=2))")
    end
end

# ===================== 主演示函数 =====================

function run_demo(;
    K::Int=3,
    N::Int=12,
    total_hours::Int=48
)
    """
    MPC演示主函数
    """
    println("="^60)
    println("BSS-MPC演示: $K 个电池, $N 小时预测, $total_hours 小时仿真")
    println("="^60)
    
    # 创建测试场景
    controller = create_test_scenario(K=K, N=N, total_hours=total_hours)
    
    # 运行MPC滚动优化
    controller = run_mpc_rolling(controller)
    
    # 分析结果
    results = analyze_mpc_results(controller)
    
    # 可视化部分结果
    visualize_power_decisions(controller, 8)
    
    println("\n" * "="^60)
    println("演示完成!")
    println("="^60)
    
    return controller, results
end

# 添加run_mpc_demo作为run_demo的别名
function run_mpc_demo(; K=3, N=12, total_hours=48)
    return run_demo(K=K, N=N, total_hours=total_hours)
end

# ===================== 导出函数 =====================

export BatteryState, create_test_scenario, run_mpc_rolling, analyze_mpc_results, 
       visualize_power_decisions, run_demo, run_mpc_demo

end # 模块结束