using JuMP, Ipopt, DataFrames, Plots, Statistics, CSVim

# 1.1 时间与电池参数
dt = 1.0  # 时间步长（小时）
N = 24    # 预测时域（24小时）
C_bat = 210.0  # 单块电池容量（kWh）

eta_ch = 0.95   # 充电效率
eta_dis = 0.95  # 放电效率
#注意：充放电效率基于实际应用中常见的电池参数合理推断

P_max = 50.0    # 最大充放电功率（kW）
SOC_init = 0.5  # 初始SOC
SOC_min = 0.2   # 最小SOC
SOC_max = 0.9   # 最大SOC

# === 新增：老化模型参数 ===(Using GP model for further development)
aging_coefficient = 0.001  # 老化惩罚系数 (USD/kW²)
# 这个系数表示每kW²功率的成本，需要根据实际电池退化成本调整

# 1.2 24小时日前市场电价数据（USD/kWh）
rho = [0.12, 0.11, 0.10, 0.10, 0.11, 0.15, 0.20, 0.18, 0.16, 0.14, 0.13, 0.14,
       0.15, 0.16, 0.17, 0.22, 0.25, 0.23, 0.19, 0.17, 0.15, 0.14, 0.13, 0.12]

# 创建JuMP模型
model = Model(Ipopt.Optimizer)
set_optimizer_attribute(model, "max_iter", 10000) # 设置最大迭代次数
set_optimizer_attribute(model, "tol", 1e-6)       # 设置收敛容差精度
set_optimizer_attribute(model, "print_level", 0)  # 输出具体精度

# 2.1 定义决策变量 - 使用单一功率变量
@variable(model, -P_max <= P[1:N] <= P_max)  # 充放电功率（负为充，正为放）
@variable(model, SOC_min <= SOC[1:N+1] <= SOC_max)  # SOC状态变量

# 2.2 约束条件
# 初始SOC约束
@constraint(model, SOC[1] == SOC_init)

# SOC更新保持简单物理
for i in 1:N
    @constraint(model, SOC[i+1] == SOC[i] - (P[i] * dt) / C_bat)
end

# 2.3 目标函数：最大化套利收益（考虑老化惩罚）
# 原目标：套利收益 = sum(P[i] * rho[i] * dt)
# 新增：老化成本 = aging_coefficient * sum(P[i]^2)
# 总目标：套利收益 - 老化成本
@objective(model, Max, 
    sum(P[i] * rho[i] * dt for i in 1:N) - 
    aging_coefficient * sum(P[i]^2 for i in 1:N)
)

# 3. 求解模型
println("开始求解带老化惩罚的优化模型...")
optimize!(model)

# 4. 结果处理和分析
status = termination_status(model)
if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED || status == MOI.ALMOST_LOCALLY_SOLVED
    println("优化求解成功！状态: ", status)
    
    # 提取结果
    P_opt = value.(P)
    SOC_opt = value.(SOC)
    total_objective = objective_value(model)
    
    # 计算各项收益和成本
    arbitrage_profit = sum(P_opt .* rho .* dt)
    aging_cost = aging_coefficient * sum(P_opt.^2)
    net_profit = arbitrage_profit - aging_cost
    
    # 计算每小时收益
    hourly_profit = P_opt .* rho .* dt
    hourly_aging_cost = aging_coefficient * P_opt.^2
    hourly_net_profit = hourly_profit - hourly_aging_cost
    
    # 整理结果
    result_df = DataFrame(
        Hour = 1:N,
        Price_USD_per_kWh = rho,
        Power_kW = round.(P_opt, digits=3),
        SOC_Start = round.(SOC_opt[1:end-1], digits=4),
        SOC_End = round.(SOC_opt[2:end], digits=4),
        Hourly_Arbitrage_USD = round.(hourly_profit, digits=3),
        Hourly_Aging_Cost_USD = round.(hourly_aging_cost, digits=3),
        Hourly_Net_Profit_USD = round.(hourly_net_profit, digits=3)
    )
    
    # 打印摘要信息
    println("\n=== 24小时套利优化结果（含老化惩罚） ===")
    println("总套利收益: ", round(arbitrage_profit, digits=2), " USD")
    println("总老化成本: ", round(aging_cost, digits=2), " USD")
    println("净收益: ", round(net_profit, digits=2), " USD")
    println("目标函数值: ", round(total_objective, digits=2), " USD")
    println("充电时段 (P < 0): ", count(x -> x < -0.1, P_opt), " 小时")
    println("放电时段 (P > 0): ", count(x -> x > 0.1, P_opt), " 小时")
    println("闲置时段 (|P| ≤ 0.1): ", count(x -> abs(x) <= 0.1, P_opt), " 小时")
    
    # 分析功率统计（老化相关）
    max_power = maximum(abs.(P_opt))
    avg_power = mean(abs.(P_opt))
    println("\n功率统计（老化分析）:")
    println("最大功率绝对值: ", round(max_power, digits=2), " kW")
    println("平均功率绝对值: ", round(avg_power, digits=2), " kW")
    
    # 分析套利策略
    charge_hours = findall(x -> x < -0.1, P_opt)
    discharge_hours = findall(x -> x > 0.1, P_opt)
    
    if !isempty(charge_hours) && !isempty(discharge_hours)
        avg_charge_price = mean(rho[i] for i in charge_hours)
        avg_discharge_price = mean(rho[i] for i in discharge_hours)
        avg_charge_power = mean(abs.(P_opt[charge_hours]))
        avg_discharge_power = mean(P_opt[discharge_hours])
        
        println("\n套利策略分析:")
        println("平均充电电价: ", round(avg_charge_price, digits=3), " USD/kWh")
        println("平均放电电价: ", round(avg_discharge_price, digits=3), " USD/kWh")
        println("平均价差: ", round(avg_discharge_price - avg_charge_price, digits=3), " USD/kWh")
        println("平均充电功率: ", round(avg_charge_power, digits=2), " kW")
        println("平均放电功率: ", round(avg_discharge_power, digits=2), " kW")
    end
    
    println("\n详细结果:")
    show(result_df, allrows=true)
    println()
    
    # 可视化结果
    p1 = plot(1:N, rho, label="Electricity Price", 
          xlabel="Hour", ylabel="Price (USD/kWh)", 
          line=:steppost, title="Electricity Price Curve", color=:blue)

    p2 = plot(1:N, P_opt, label="Charge/Discharge Power", 
          xlabel="Hour", ylabel="Power (kW)",
          line=:steppost, title="Power Strategy (with Aging Consideration)", color=:red)
    hline!([0], label="Zero", color=:black, linestyle=:dash, alpha=0.5)

    p3 = plot(0:N, SOC_opt, label="SOC", 
          xlabel="Hour", ylabel="SOC", 
          title="SOC Evolution", color=:green)
    hline!([SOC_min, SOC_max], label=["SOC_min" "SOC_max"], 
           color=[:red :red], linestyle=:dash, alpha=0.5)
    
    # 新增：老化成本可视化
    p4 = plot(1:N, hourly_aging_cost, label="Aging Cost", 
          xlabel="Hour", ylabel="Cost (USD)", 
          line=:steppost, title="Hourly Aging Cost", color=:orange)

    plot(p1, p2, p3, p4, layout=(4,1), size=(800, 1000))
    
    # 保存结果
    savefig("arbitrage_results_with_aging.png")
    println("\n结果图已保存为 arbitrage_results_with_aging.png")
    
    # 保存详细结果到CSV
    CSV.write("battery_arbitrage_with_aging.csv", result_df)
    println("详细结果已保存为 battery_arbitrage_with_aging.csv")
    
else
    println("优化求解失败，原因: ", status)
end