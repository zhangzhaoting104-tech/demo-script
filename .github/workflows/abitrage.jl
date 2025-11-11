using JuMP, Ipopt, DataFrames

# 1.1 时间与电池参数（根据需求调整）
dt = 1  # 时间步长（小时）
N = 24  # 预测时域（24小时）
C_bat = 210.0  # 单块电池容量（kWh，参考文章NIO换电站单电池容量）
eta_ch = 0.95  # 充电效率
eta_dis = 0.95  # 放电效率
P_max = 50.0  # 最大充放电功率（kW，负为充，正为放）
SOC_init = 0.5  # 初始SOC
SOC_min = 0.2  # 最小SOC（避免过放）
SOC_max = 0.9  # 最大SOC（避免过充）

# 1.2 24小时日前市场电价数据（单位USD/kWh）
# 模拟电价：凌晨低、早高峰高、晚高峰高（符合“电价波动套利”逻辑,ai生成未使用PJM真实数据）
rho = [0.12, 0.11, 0.10, 0.10, 0.11, 0.15, 0.20, 0.18, 0.16, 0.14, 0.13, 0.14,
       0.15, 0.16, 0.17, 0.22, 0.25, 0.23, 0.19, 0.17, 0.15, 0.14, 0.13, 0.12]

# 创建JuMP模型，指定IPOPT求解器（无整数变量，用非线性求解器即可）
model = Model(Ipopt.Optimizer)
# 关键修改：创建模型时指定IPOPT求解器，并设置最大迭代次数（如5000次，可根据需要调整）
model = Model(()->Ipopt.Optimizer(
    "max_iter" => 5000,  # 核心参数：最大迭代次数，默认1000，此处改为5000
    "tol" => 1e-6,       # 可选：松弛收敛精度（默认1e-8），适当放宽可加快收敛
    "print_level" => 3   # 可选：打印求解过程（1=少，5=多），便于排查问题
))

# 2.1 定义决策变量（控制输入P_i + 状态变量SOC_i）
@variable(model, P[1:N])  # 每小时充放电功率（kW）
@variable(model, SOC[1:N+1])  # SOC_i对应第i步结束时的状态（共25个点：1→24+1）

# 2.2 约束条件（对应文章中“s.t.”部分）
# 初始SOC约束
@constraint(model, SOC[1] == SOC_init)

# 状态转移约束（逐时间步更新SOC，分充电/放电逻辑）
for i in 1:N
    # 充电场景（P[i] < 0：购电，功率取绝对值）
    @constraint(model, SOC[i+1] >= SOC[i] + (abs(P[i]) * dt * eta_ch) / C_bat)
    # 放电场景（P[i] > 0：售电，需除以放电效率）
    @constraint(model, SOC[i+1] <= SOC[i] - (P[i] * dt) / (eta_dis * C_bat))
    # 功率边界约束
    @constraint(model, P[i] >= -P_max)
    @constraint(model, P[i] <= P_max)
    # SOC安全约束（每步结束后SOC需在范围内）
    @constraint(model, SOC[i+1] >= SOC_min)
    @constraint(model, SOC[i+1] <= SOC_max)
end

# 2.3 目标函数（最大化24小时套利净收益，对应文章中“max R”）
# 收益单位：USD（功率kW * 电价USD/kWh * 时间h = USD）
@objective(model, Max, sum(P[i] * rho[i] * dt for i in 1:N))

# 求解模型（IPOPT会自动处理非线性约束）
optimize!(model)

# 3.1 提取结果（判断求解状态，确保可行）
if termination_status(model) == MOI.OPTIMAL
    println("优化求解成功！")
    # 提取最优功率和SOC
    P_opt = value.(P)  # 最优充放电功率（kW）
    SOC_opt = value.(SOC)  # 最优SOC轨迹
    total_profit = objective_value(model)  # 总套利收益（USD）
    
    # 3.2 整理结果为DataFrame（便于查看和后续分析）
    result_df = DataFrame(
        Hour = 1:N,  # 第1-24小时
        Price_USD_per_kWh = rho,
        Opt_Power_kW = round.(P_opt, digits=2),  # 四舍五入保留2位小数
        SOC_End = round.(SOC_opt[2:end], digits=3),  # 每小时结束时的SOC
        Hourly_Profit_USD = round.(P_opt .* rho .* dt, digits=2)  # 每小时收益
    )
    # 打印结果
    println("\n24小时套利优化结果：")
    println(result_df)
    println("\n总套利收益：", round(total_profit, digits=2), " USD")
else
    println("优化求解失败，原因：", termination_status(model))
end