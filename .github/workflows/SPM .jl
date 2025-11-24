using JuMP, Ipopt, DataFrames, Plots, Statistics, CSV

# ==========================================
# 1. 参数定义
# ==========================================

# 1.1 仿真时间设置
dt_h = 1.0              # 时间步长 (小时)
dt_s = dt_h * 3600.0    # 时间步长 (秒)
N = 24                  # 预测时域 (24小时)

# 1.2 电池包系统参数 
Pack_Capacity_kWh = 210.0       # 电池总容量 (kWh)
P_max_kW = 50.0                 # 最大充放电功率 (kW)
SOC_init = 0.5                  # 初始荷电状态
SOC_min = 0.1                   # 最小SOC限制 (保护电池)
SOC_max = 0.95                  # 最大SOC限制 (防止过充)

# 1.3 电池参数 
Cell_Capacity_Ah = 2.5          # 单体电芯容量 (Ah)
Cell_Nominal_V = 3.3            # 单体额定电压 (V)
Cell_Energy_Wh = Cell_Capacity_Ah * Cell_Nominal_V # 单体能量 ~8.25 Wh

# 计算电池包包含的电芯数量
Num_Cells = (Pack_Capacity_kWh * 1000) / Cell_Energy_Wh
println("系统等效电芯数量: ", round(Num_Cells, digits=0))

# 1.4 电化学参数 (基于单粒子模型SPM)
F = 96485.0       # 法拉第常数 (库仑/摩尔)
R = 8.314         # 气体常数 (焦耳/(摩尔·K))
T = 298.15        # 温度 (K) - 假设恒温25°C

# 负极参数 (石墨材料)
R_n = 5.0e-6      # 负极颗粒半径 (米)
D_n = 5.0e-14     # 负极扩散系数 (平方米/秒) - 调整为功率型电池
A_n = 0.35        # 负极有效表面积 (平方米)
k_n = 2.0e-10     # 反应速率常数 (提高动力学性能)
c_e = 1000.0      # 电解液浓度 (摩尔/立方米)
c_n_max = 26000.0 # 负极最大锂浓度 (摩尔/立方米)

# SEI膜生长参数 (关键老化机制)
k_SEI = 1.5e-12   # SEI反应动力学常数
U_ref = 0.4       # SEI反应起始电位 (V)
rho_SEI = 2000.0  # SEI膜密度
M_SEI = 0.1       # SEI膜分子量

# 1.5 经济参数校准
# 计算逻辑：
# - 电池包成本约 $30,000
# - 寿命终止定义为容量衰减20%
# - 系统总容量 = 电芯数 × 2.5Ah ≈ 63,637Ah
# - 允许损失容量 = 63,637 × 20% ≈ 12,727Ah
# - 每损失1Ah的成本 = $30,000 / 12,727 ≈ $2.35/Ah
# - 设置稍高惩罚值，平衡经济收益与电池寿命
aging_penalty_factor = 5.0  # 老化惩罚系数 (美元/Ah损失)

# 1.6 24小时电价数据 (美元/kWh)
rho = [0.12, 0.11, 0.10, 0.10, 0.11, 0.15, 0.20, 0.18, 0.16, 0.14, 0.13, 0.14,
       0.15, 0.16, 0.17, 0.22, 0.25, 0.23, 0.19, 0.17, 0.15, 0.14, 0.13, 0.12]

# ==========================================
# 2. 辅助函数定义
# ==========================================

# 负极开路电压函数 (基于SOC的拟合关系)
function OCV_n(theta)
    return 0.7222 + 0.1387 * theta + 0.029 * theta^0.5 - 
           0.0172/theta + 0.0019/theta^1.5 + 
           0.2808 * exp(0.9 - 15 * theta) - 
           0.7984 * exp(0.4465 * theta - 0.4108)
end

# ==========================================
# 3. 优化模型构建
# ==========================================

# 创建JuMP优化模型，使用Ipopt求解器
model = Model(Ipopt.Optimizer)

# 设置求解器参数
set_optimizer_attribute(model, "max_iter", 3000)  # 最大迭代次数
set_optimizer_attribute(model, "tol", 1e-4)       # 收敛容差
set_optimizer_attribute(model, "print_level", 5)  # 输出详细程度

# --- 3.1 定义决策变量 ---

# 电池功率决策 (充放电)
@variable(model, -P_max_kW <= P_pack[1:N] <= P_max_kW, start=0.0) 

# 电池荷电状态序列
@variable(model, SOC_min <= SOC[1:N+1] <= SOC_max, start=SOC_init)

# --- 3.2 电化学模型中间变量 ---

# 电芯电流 (转换为单体级别)
@variable(model, -20.0 <= I_cell[1:N] <= 20.0, start=0.0)    

# 负极电位
@variable(model, phi_n[1:N], start=0.5)           

# 过电位
@variable(model, eta_n[1:N], start=0.0)           

# 负极锂浓度归一化值
@variable(model, 0.01 <= theta_n[1:N] <= 0.99, start=0.35) 

# SEI副反应电流 (必须为负值)
@variable(model, -1e-4 <= i_SEI[1:N] <= 0, start=-1e-10)      

# --- 3.3 初始条件约束 ---
@constraint(model, SOC[1] == SOC_init)

# --- 3.4 系统动力学与电化学约束 ---
for k in 1:N
    # (1) 系统功率到电芯电流的转换
    @constraint(model, I_cell[k] == -(P_pack[k] * 1000.0) / (Num_Cells * Cell_Nominal_V))
    
    # (2) SOC状态更新 (基于能量平衡)
    @constraint(model, SOC[k+1] == SOC[k] - (P_pack[k] * dt_h) / Pack_Capacity_kWh)

    # (3) SOC与负极锂浓度的映射关系
    @constraint(model, theta_n[k] == 0.0 + 0.7 * SOC[k]) 

    # (4) Butler-Volmer动力学方程
    # 描述电化学反应速率与过电位的关系
    @NLconstraint(model, I_cell[k] == 0.115 * sqrt(theta_n[k] * (1 - theta_n[k])) * sinh( 19.46 * eta_n[k] ))

    # (5) 负极电位计算
    # 开路电压 + 过电位
    @NLconstraint(model, phi_n[k] == (0.1 + 0.1 * theta_n[k] + 0.6 * exp(-10 * theta_n[k])) + eta_n[k])

    # (6) SEI副反应电流 (核心老化模型)
    # 描述SEI膜生长速率与负极电位的关系
    @NLconstraint(model, i_SEI[k] == -5.25e-13 * exp( -38.92 * (phi_n[k] - U_ref) ))
end

# ==========================================
# 4. 目标函数
# ==========================================

# 最大化总收益 = 电能套利收益 - 电池老化成本
@NLobjective(model, Max, 
    # 电能套利收益：功率 × 电价 × 时间
    sum(P_pack[i] * rho[i] * dt_h for i in 1:N) 
    # 电池老化成本：SEI电流 × 时间 × 电芯数 × 惩罚系数
    + sum(i_SEI[i] * dt_s * Num_Cells * aging_penalty_factor / 3600.0 for i in 1:N) 
)

# ==========================================
# 5. 模型求解与结果分析
# ==========================================

println("开始求解 (校准参数版)...")
println("老化惩罚因子: ", aging_penalty_factor, " USD/Ah")

# 执行优化计算
optimize!(model)

# 检查求解状态
status = termination_status(model)
if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
    println("优化成功！状态: ", status)
    
    # 提取优化结果
    P_opt = value.(P_pack)
    SOC_opt = value.(SOC)
    i_SEI_opt = value.(i_SEI)
    phi_n_opt = value.(phi_n)
    
    # 计算经济效益
    revenue = sum(P_opt .* rho .* dt_h)  # 总电费收入
    total_capacity_loss_Ah = sum(abs.(i_SEI_opt) .* dt_s) * Num_Cells / 3600.0  # 总容量损失
    aging_cost = total_capacity_loss_Ah * aging_penalty_factor  # 老化成本
    net_profit = revenue - aging_cost  # 净利润
    
    # 输出关键结果
    println("\n" * "="^50)
    println("基于SPM模型的套利优化结果")
    println("="^50)
    println("总电费收入:       \$", round(revenue, digits=2))
    println("电池老化成本:     \$", round(aging_cost, digits=2))
    println("净利润:          \$", round(net_profit, digits=2))
    println("总容量损失:      ", round(total_capacity_loss_Ah, digits=4), " Ah")
    
    # 策略效果评估
    if revenue > 0.1
        println("策略成功：电池有效参与电能套利！")
    else
        println("策略警告：套利收益不足，请检查电价设置。")
    end

    # 生成详细结果表格
    df = DataFrame(
        小时 = 1:N,
        电价 = rho,
        功率_kW = round.(P_opt, digits=3),
        荷电状态 = round.(SOC_opt[1:end-1], digits=3),
        负极电位_V = round.(phi_n_opt, digits=4),
        SEI电流_nA = round.(i_SEI_opt .* 1e9, digits=2)
    )
    
    println("\n详细小时数据:")
    show(df, allrows=true)
    
    # 保存结果到CSV文件
    CSV.write("spm_arbitrage_results.csv", df)
    
    # 生成结果可视化图表
    p1 = plot(1:N, rho, title="Electricity Price (\$/kWh)", label="", line=:steppost, color=:blue)
    p2 = plot(1:N, P_opt, title="Power (kW)", label="Power", line=:steppost, color=:red)
    hline!(p2, [0], color=:black, linestyle=:dash, label="Zero Line")
    p3 = plot(1:N, phi_n_opt, title="Negative (V)", label="phi_n", color=:purple)
    hline!(p3, [U_ref], label="SEI Onset", color=:red, linestyle=:dash)
    p4 = plot(1:N, abs.(i_SEI_opt) .* Num_Cells, title="Aging Rate(A)", label="Current_Loss", color=:orange)
    
    # 组合图表并保存
    final_plot = plot(p1, p2, p3, p4, layout=(4,1), size=(800, 1000))
    savefig(final_plot, "spm_optimization_plot.png")
    println("\n优化结果图表已保存")

else
    println("求解失败，状态: ", status)
end