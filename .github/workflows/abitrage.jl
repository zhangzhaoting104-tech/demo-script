using JuMP, Ipopt, DataFrames, Plots, Statistics, CSV

# 1.1 时间与电池参数
dt = 1.0  # 时间步长（小时）
N = 24    # 预测时域（24小时）
C_bat = 210.0  # 单块电池容量（kWh）
P_max = 50.0    # 最大充放电功率（kW）
SOC_init = 0.5  # 初始SOC
SOC_min = 0.2   # 最小SOC
SOC_max = 0.9   # 最大SOC

# 线性老化惩罚参数（USD/kWh）- 每充放电1kWh的老化成本
linear_aging_coefficient = 0.015  # ai推荐起始值0.015

# 1.2 24小时日前市场电价数据（USD/kWh）
rho = [0.12, 0.11, 0.10, 0.10, 0.11, 0.15, 0.20, 0.18, 0.16, 0.14, 0.13, 0.14,
       0.15, 0.16, 0.17, 0.22, 0.25, 0.23, 0.19, 0.17, 0.15, 0.14, 0.13, 0.12]

# 创建JuMP模型
model = Model(Ipopt.Optimizer)
set_optimizer_attribute(model, "max_iter", 10000)
set_optimizer_attribute(model, "tol", 1e-6)
set_optimizer_attribute(model, "print_level", 0)

# 2.1 定义决策变量
@variable(model, -P_max <= P[1:N] <= P_max)  # 充放电功率（负为充，正为放）
@variable(model, SOC_min <= SOC[1:N+1] <= SOC_max)  # SOC状态变量

# 新增：功率绝对值辅助变量（避免在目标函数中直接使用abs）
@variable(model, P_abs[1:N] >= 0)  # 功率的绝对值

# 2.2 约束条件
# 初始SOC约束
@constraint(model, SOC[1] == SOC_init)

# SOC动态约束,注意负值为充电，公式更新符号为-
for i in 1:N
    @constraint(model, SOC[i+1] == SOC[i] - (P[i] * dt) / C_bat)
end

# 功率绝对值约束（线性化abs函数，确保P_abs[i]=|P[i]|）
for i in 1:N
    @constraint(model, P_abs[i] >= P[i])   # P_abs >= P
    @constraint(model, P_abs[i] >= -P[i])  # P_abs >= -P
end

# 2.3 目标函数：最大化套利收益（考虑线性老化惩罚）
@objective(model, Max, 
    sum(P[i] * rho[i] * dt for i in 1:N) - linear_aging_coefficient * sum(P_abs[i] * dt for i in 1:N)
)

# 3. 求解模型
println("开始求解带线性老化惩罚的优化模型喵(=^･ω･^=)")
println("线性老化系数: ", linear_aging_coefficient, " USD/kWh")
println("正在求解，请稍候喵(=^･ω･^=)")

optimize!(model)

# 4. 结果处理和分析
status = termination_status(model)
if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED || status == MOI.ALMOST_LOCALLY_SOLVED
    println("优化求解成功喵(=^･ω･^=)！状态: ", status)
    
    # 提取结果
    P_opt = value.(P)
    P_abs_opt = value.(P_abs)
    SOC_opt = value.(SOC)
    total_objective = objective_value(model)
    
    # 计算各项收益和成本
    arbitrage_profit = sum(P_opt .* rho .* dt)
    total_energy_throughput = sum(P_abs_opt) * dt  # 总能量吞吐量
    aging_cost = linear_aging_coefficient * total_energy_throughput
    net_profit = arbitrage_profit - aging_cost
    
    # 计算每小时收益和成本
    hourly_profit = P_opt .* rho .* dt
    hourly_energy_throughput = P_abs_opt .* dt
    hourly_aging_cost = linear_aging_coefficient * hourly_energy_throughput
    hourly_net_profit = hourly_profit - hourly_aging_cost
    
    # 整理结果
    result_df = DataFrame(
        Hour = 1:N,
        Price_USD_per_kWh = rho,
        Power_kW = round.(P_opt, digits=3),
        Power_Abs_kW = round.(P_abs_opt, digits=3),
        SOC_Start = round.(SOC_opt[1:end-1], digits=4),
        SOC_End = round.(SOC_opt[2:end], digits=4),
        Energy_Throughput_kWh = round.(hourly_energy_throughput, digits=3),
        Hourly_Arbitrage_USD = round.(hourly_profit, digits=3),
        Hourly_Aging_Cost_USD = round.(hourly_aging_cost, digits=3),
        Hourly_Net_Profit_USD = round.(hourly_net_profit, digits=3)
    )
    
    # 打印摘要信息
    println("\n" * "="^60)
    println("24-Hour Arbitrage Optimization Results (Linear Aging Penalty)")
    println("="^60)
    println("Linear Aging Coefficient: ", linear_aging_coefficient, " USD/kWh")
    println("Total Arbitrage Profit: ", round(arbitrage_profit, digits=2), " USD")
    println("Total Energy Throughput: ", round(total_energy_throughput, digits=2), " kWh")
    println("Total Aging Cost: ", round(aging_cost, digits=2), " USD")
    println("Net Profit: ", round(net_profit, digits=2), " USD")
    println("Objective Value: ", round(total_objective, digits=2), " USD")
    
    # 功率行为分析
    charge_hours = findall(x -> x < -0.1, P_opt)
    discharge_hours = findall(x -> x > 0.1, P_opt)
    idle_hours = findall(x -> abs(x) <= 0.1, P_opt)
    
    println("\nOperation Analysis:")
    println("Charging Hours (P < -0.1): ", length(charge_hours))
    println("Discharging Hours (P > 0.1): ", length(discharge_hours))
    println("Idle Hours (|P| ≤ 0.1): ", length(idle_hours))
    
    # 功率统计
    max_charge_power = isempty(charge_hours) ? 0.0 : minimum(P_opt[charge_hours])
    max_discharge_power = isempty(discharge_hours) ? 0.0 : maximum(P_opt[discharge_hours])
    avg_power_abs = mean(P_abs_opt)
    
    println("\nPower Statistics:")
    println("Max Charging Power: ", round(max_charge_power, digits=2), " kW")
    println("Max Discharging Power: ", round(max_discharge_power, digits=2), " kW")
    println("Average Absolute Power: ", round(avg_power_abs, digits=2), " kW")
    
    # 套利策略分析
    if !isempty(charge_hours) && !isempty(discharge_hours)
        avg_charge_price = mean(rho[i] for i in charge_hours)
        avg_discharge_price = mean(rho[i] for i in discharge_hours)
        price_spread = avg_discharge_price - avg_charge_price
        
        # 考虑老化成本的有效价差
        effective_spread_required = 2 * linear_aging_coefficient  # 往返老化成本
        effective_price_spread = price_spread - effective_spread_required
        
        println("\nArbitrage Strategy Analysis:")
        println("Average Charge Price: ", round(avg_charge_price, digits=3), " USD/kWh")
        println("Average Discharge Price: ", round(avg_discharge_price, digits=3), " USD/kWh")
        println("Nominal Price Spread: ", round(price_spread, digits=3), " USD/kWh")
        println("Required Min Effective Spread: ", round(effective_spread_required, digits=3), " USD/kWh")
        println("Effective Price Spread: ", round(effective_price_spread, digits=3), " USD/kWh")
        
        if effective_price_spread > 0
            println("✅ Arbitrage strategy is economically viable")
        else
            println("⚠️  Arbitrage strategy economic viability is questionable")
        end
    end
    
    # 老化成本分析
    println("\nAging Cost Analysis:")
    println("Unit Aging Cost: ", linear_aging_coefficient, " USD/kWh")
    aging_cost_ratio = aging_cost/arbitrage_profit*100
    println("Aging Cost Ratio: ", round(aging_cost_ratio, digits=1), "%")
    println("Aging Cost per Dollar Profit: ", round(aging_cost/arbitrage_profit, digits=3), " USD")
    
    # 识别高成本操作
    high_cost_operations = findall(x -> x > 0.1, hourly_aging_cost)
    if !isempty(high_cost_operations)
        println("High Aging Cost Hours: ", high_cost_operations)
    end
    
    println("\nFirst 10 Hours Detailed Results:")
    show(first(result_df, 10), allrows=true)
    println()
    
    # 可视化结果 - 使用英文标签避免字体问题
    p1 = plot(1:N, rho, label="Electricity Price", 
          xlabel="Hour", ylabel="Price (USD/kWh)", 
          line=:steppost, title="Electricity Price Curve", color=:blue,
          ylim=(0.09, 0.27))

    p2 = plot(1:N, P_opt, label="Charge/Discharge Power", 
          xlabel="Hour", ylabel="Power (kW)",
          line=:steppost, title="Power Strategy (Linear Aging Penalty)", color=:red)
    hline!([0], label="Zero", color=:black, linestyle=:dash, alpha=0.5)
    ylims!(-P_max-5, P_max+5)

    p3 = plot(0:N, SOC_opt, label="SOC", 
          xlabel="Hour", ylabel="SOC", 
          title="SOC Evolution", color=:green)
    hline!([SOC_min, SOC_max], label=["SOC_min" "SOC_max"], 
           color=[:red :red], linestyle=:dash, alpha=0.5)
    ylims!(SOC_min-0.05, SOC_max+0.05)
    
    # 老化成本可视化
    p4 = plot(1:N, hourly_aging_cost, label="Hourly Aging Cost", 
          xlabel="Hour", ylabel="Cost (USD)", 
          line=:steppost, title="Hourly Aging Cost", color=:orange)
    hline!([mean(hourly_aging_cost)], label="Average", color=:red, linestyle=:dash)

    # 净收益可视化
    p5 = plot(1:N, hourly_net_profit, label="Hourly Net Profit", 
          xlabel="Hour", ylabel="Profit (USD)", 
          line=:steppost, title="Hourly Net Profit", color=:purple)
    hline!([0], label="Break-even", color=:black, linestyle=:dash, alpha=0.5)
    
    plot(p1, p2, p3, p4, p5, layout=(5,1), size=(800, 1200))
    
    # 保存结果
    savefig("arbitrage_results_linear_aging.png")
    println("\nResults plot saved as arbitrage_results_linear_aging.png")
    
    # 保存详细结果到CSV
    CSV.write("battery_arbitrage_linear_aging.csv", result_df)
    println("Detailed results saved as battery_arbitrage_linear_aging.csv")
    
    # 输出建议
    println("\n" * "="^40)
    println("Parameter Tuning Suggestions")
    println("="^40)
    if aging_cost_ratio > 30
        println("Suggestion: Aging cost ratio too high (>30%), consider reducing coefficient to 0.01 USD/kWh")
    elseif aging_cost_ratio < 10
        println("Suggestion: Aging cost ratio low (<10%), consider increasing to 0.02 USD/kWh for better battery protection")
    else
        println("Suggestion: Current coefficient ", linear_aging_coefficient, " USD/kWh is well balanced")
    end
    
    println("\nAlternative parameters to try:")
    println("• Aggressive strategy: linear_aging_coefficient = 0.005")
    println("• Balanced strategy: linear_aging_coefficient = 0.015 (current)")
    println("• Conservative strategy: linear_aging_coefficient = 0.025")
    
else
    println("Optimization failed, reason: ", status)
    println("\nDebugging suggestions:")
    println("1. Try reducing the linear aging coefficient")
    println("2. Check if constraints are too tight")
    println("3. Try different initial values")
end