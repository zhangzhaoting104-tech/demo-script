using JuMP, Ipopt, DataFrames, Plots, Statistics

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

# 1.2 24小时日前市场电价数据（USD/kWh）
rho = [0.12, 0.11, 0.10, 0.10, 0.11, 0.15, 0.20, 0.18, 0.16, 0.14, 0.13, 0.14,
       0.15, 0.16, 0.17, 0.22, 0.25, 0.23, 0.19, 0.17, 0.15, 0.14, 0.13, 0.12]

# 创建JuMP模型
model = Model(Ipopt.Optimizer)
set_optimizer_attribute(model, "max_iter", 10000) # 设置最大迭代次数
set_optimizer_attribute(model, "tol", 1e-6)       # 设置收敛容差精度
set_optimizer_attribute(model, "print_level", 0)  # 减少输出

# 2.1 定义决策变量 - 使用单一功率变量
@variable(model, -P_max <= P[1:N] <= P_max)  # 充放电功率（负为充，正为放）
@variable(model, SOC_min <= SOC[1:N+1] <= SOC_max)  # SOC状态变量

# 2.2 约束条件
# 初始SOC约束
@constraint(model, SOC[1] == SOC_init)

# 状态转移约束 - 使用近似线性处理
for i in 1:N
    # 使用平均效率近似，避免非线性
    eta_avg = (eta_ch + 1/eta_dis) / 2  # 平均效率
    @constraint(model, SOC[i+1] == SOC[i] + (P[i] * dt * eta_avg) / C_bat)
end

# 2.3 目标函数：最大化套利收益
@objective(model, Max, sum(P[i] * rho[i] * dt for i in 1:N))

# 3. 求解模型
println("开始求解...")
optimize!(model)

# 4. 结果处理和分析
status = termination_status(model)
if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED || status == MOI.ALMOST_LOCALLY_SOLVED
    println("优化求解成功！状态: ", status)
    
    # 提取结果
    P_opt = value.(P)
    SOC_opt = value.(SOC)
    total_profit = objective_value(model)
    
    # 计算每小时收益
    hourly_profit = P_opt .* rho .* dt
    
    # 整理结果
    result_df = DataFrame(
        Hour = 1:N,
        Price_USD_per_kWh = rho,
        Power_kW = round.(P_opt, digits=3),
        SOC_Start = round.(SOC_opt[1:end-1], digits=4),
        SOC_End = round.(SOC_opt[2:end], digits=4),
        Hourly_Profit_USD = round.(hourly_profit, digits=3)
    )
    
    # 打印摘要信息
    println("\n=== 24小时套利优化结果 ===")
    println("总套利收益: ", round(total_profit, digits=2), " USD")
    println("充电时段 (P < 0): ", count(x -> x < -0.1, P_opt), " 小时")
    println("放电时段 (P > 0): ", count(x -> x > 0.1, P_opt), " 小时")
    println("闲置时段 (|P| ≤ 0.1): ", count(x -> abs(x) <= 0.1, P_opt), " 小时")
    
    # 分析套利策略
    charge_hours = findall(x -> x < -0.1, P_opt)
    discharge_hours = findall(x -> x > 0.1, P_opt)
    
    if !isempty(charge_hours) && !isempty(discharge_hours)
        avg_charge_price = mean(rho[i] for i in charge_hours)
        avg_discharge_price = mean(rho[i] for i in discharge_hours)
        println("平均充电电价: ", round(avg_charge_price, digits=3), " USD/kWh")
        println("平均放电电价: ", round(avg_discharge_price, digits=3), " USD/kWh")
        println("平均价差: ", round(avg_discharge_price - avg_charge_price, digits=3), " USD/kWh")
    end
    
    println("\n详细结果:")
    show(result_df, allrows=true)
    println()
    
    # 可视化结果
    p1 = plot(1:N, rho, label="电价", xlabel="小时", ylabel="电价 (USD/kWh)", 
             line=:steppost, title="电价曲线", color=:blue, legend=:topleft)
    
    p2 = plot(1:N, P_opt, label="充放电功率", xlabel="小时", ylabel="功率 (kW)",
             line=:steppost, title="充放电策略", color=:red, legend=:topleft)
    hline!([0], label="", color=:black, linestyle=:dash, alpha=0.5)
    
    p3 = plot(0:N, SOC_opt, label="SOC", xlabel="小时", ylabel="SOC", 
             title="SOC演化", color=:green, legend=:bottomright)
    hline!([SOC_min, SOC_max], label=["SOC_min" "SOC_max"], 
           color=[:red :red], linestyle=:dash, alpha=0.5)
    
    plot(p1, p2, p3, layout=(3,1), size=(800, 800))
    
    # 保存结果
    savefig("arbitrage_results.png")
    println("\n结果图已保存为 arbitrage_results.png")
    
else
    println("优化求解失败，原因: ", status)
end