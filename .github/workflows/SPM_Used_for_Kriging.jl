using Random, LinearAlgebra, CSV, DataFrames

# SPM电池模型参数
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
    SPMParameters(
        # 扩散系数 - 表1中的 D_p, D_n
        1.736e-14,   # D_p (m²/s) - 正极扩散系数
        8.256e-14,   # D_n (m²/s) - 负极扩散系数
        
        # 粒子半径 - 表1中的 R_p, R_n  
        1.637e-7,    # R_p (m) - 正极粒子半径
        3.596e-6,    # R_n (m) - 负极粒子半径
        
        # 最大浓度 - 表1中的 C_p_max, C_n_max
        10350,   # C_p_max (mol/m³) - 正极最大浓度
        29480,   # C_n_max (mol/m³) - 负极最大浓度
        
        # 几何参数 - 表1中的 a_p, a_n, l_p, l_n
        8800000.0,  # a_p (m²/m³) - 正极表面积体积比
        318000.0,  # a_n (m²/m³) - 负极表面积体积比
        6.521e-5,   # l_p (m) - 正极厚度
        2.885e-5,   # l_n (m) - 负极厚度
        
        # 反应速率常数 - 表1中的 k_p, k_n
        1.168e-12, # k_p (m².⁵/mol⁰.⁵·s) - 正极反应速率
        9.016e-12, # k_n (m².⁵/mol⁰.⁵·s) - 负极反应速率
        
        # SEI相关参数 - 表1中的参数
        1.5e-12,   # k_SEI (A/m²) - SEI反应速率常数
        5.0e-6,    # κ_SEI (S/m) - SEI离子电导率
        0.073,     # M_SEI (kg/mol) - SEI分子量
        2100.0,    # ρ_SEI (kg/m³) - SEI密度
        0.4,       # U_ref (V) - SEI反应参考电位
        
        # 常数 - 表1中的 F, R, T
        96485.0,   # F (C/mol) - 法拉第常数
        8.314,     # R (J/mol·K) - 气体常数
        298.15,    # T (K) - 温度
        
        # 电解质浓度 - 表1中的 C_e
        1042.0     # C_e (mol/m³) - 电解质浓度
    )
end

# 更接近实际LiFePO4电池的OCV函数
function OCV_positive(θ_p)
    # LiFePO4的平坦平台特性
    if θ_p < 0.05
        return 3.30 + 2.0 * θ_p
    elseif θ_p < 0.95  
        return 3.40  # 主要平坦平台
    else
        return 3.40 + 0.5 * (θ_p - 0.95)
    end
end

function OCV_negative(θ_n)
    # 石墨负极的典型范围
    return 0.12 + 0.15 * θ_n + 0.05 * θ_n^2
end

# 生成随机状态和输入
function generate_random_state_input(params::SPMParameters)
    # 生成随机状态 [C_p_avg, C_n_avg, δ_SEI, c_f]
    state = [
        rand(0.2:0.01:0.8) * params.C_p_max,  # C_p_avg: 10%-80% of max
        rand(0.2:0.01:0.8) * params.C_n_max,  # C_n_avg: 10%-80% of max
        rand(0.0:1e-10:5e-8),                 # δ_SEI: 0 to 100 nm
        rand(0.0:0.1:50.0)                    # c_f: 0 to 50 Ah capacity fade
    ]
    
    # 生成随机输入 (功率，单位: kW)
    input = rand(-50.0:0.1:50.0)  # -50 kW to +50 kW
    
    return state, input
end


# # 完整的SPM状态增量计算函数
function calculate_SPM_state_increment(params::SPMParameters, state, input, dt=1.0)
    """
    完整的SPM状态增量计算,包含Butler-Volmer动力学
    """
    C_p_avg, C_n_avg, δ_SEI, c_f = state
    P = input
    
    # 迭代求解电流和电位
    converged = false
    count_iter = 0
    max_iter = 20
    tol = 1e-6
    i_guess = 50  # 初始电流猜测
  
    for iter in 1:max_iter
        # 1. 浓度计算
        C_p_surf = C_p_avg - (params.R_p / 5) * i_guess / (params.F * params.D_p * params.a_p * params.l_p)
        C_n_surf = C_n_avg + (params.R_n / 5) * i_guess / (params.F * params.D_n * params.a_n * params.l_n)
        # 确保浓度在合理范围内
        C_p_surf = max(0.01 * params.C_p_max, min(0.99 * params.C_p_max, C_p_surf))
        C_n_surf = max(0.01 * params.C_n_max, min(0.99 * params.C_n_max, C_n_surf))
        
        # 2. OCV计算
        θ_p = C_p_surf / params.C_p_max #公式3.5
        θ_n = C_n_surf / params.C_n_max
        U_p = OCV_positive(θ_p)         #Genetic论文OCV函数
        U_n = OCV_negative(θ_n)
        
        # 3. Butler-Volmer动力学
        φ_p_guess = U_p + 0.1
        J_p = 2 * params.k_p * params.C_e^0.5 * 
              (params.C_p_max - C_p_surf)^0.5 * (C_p_surf)^0.5 *
              sinh(0.5 * params.F * (φ_p_guess - U_p) / (params.R * params.T))
            #含义：正极单位体积的反应电流（左侧），由反应速率常数（\(k_p\)）、电解质浓度（\(C_e\)）、表面锂浓度（\(C_{p}^{surf}\)）、过电位（\(\phi_p - U_p\)）共同决定
        #论文公式3.7

        i_p_calc = J_p * params.a_p * params.F * params.l_p
        # 正极总电流 = 电流密度 × 表面积 × 法拉第常数 × 厚度
        
        φ_n_guess = U_n - 0.1
        J_n = 2 * params.k_n * params.C_e^0.5 * 
              (params.C_n_max - C_n_surf)^0.5 * (C_n_surf)^0.5 *
              sinh(0.5 * params.F * 
              (φ_n_guess - U_n + (δ_SEI / params.κ_SEI) * (i_guess / (params.a_n * params.l_n))) / 
              (params.R * params.T))
              # 负极反应电流密度，来自论文式(3.8)，包含SEI电阻影响

        i_n_calc = -J_n * params.a_n * params.F * params.l_n
        # 负极总电流 = 电流密度 × 表面积 × 法拉第常数 × 厚度
        count_iter += 1
        # 4. 电流连续性检查
        i_new = (i_p_calc + i_n_calc) / 2
        
        if abs(i_new - i_guess) < tol
            converged = true
            i_guess = i_new
            break
        end
        i_guess = i_new
    end
    
    println("迭代次数: $count_iter, 收敛: $converged, 最终电流: $i_guess A")

    if !converged
        println("迭代未收敛，使用最后的电流猜测值")
        i_guess = 50
    end

    # 使用最终电流重新计算相关量
    C_p_surf = C_p_avg - (params.R_p / 5) * i_guess / (params.F * params.D_p * params.a_p * params.l_p)
    C_n_surf = C_n_avg + (params.R_n / 5) * i_guess / (params.F * params.D_n * params.a_n * params.l_n)
    θ_p = C_p_surf / params.C_p_max
    θ_n = C_n_surf / params.C_n_max
    U_p = OCV_positive(θ_p)
    U_n = OCV_negative(θ_n)
    
    # SEI反应
    η_SEI = U_n - params.U_ref + δ_SEI / params.κ_SEI * i_guess / (params.a_n * params.l_n)
    i_SEI = params.a_n * params.l_n * params.k_SEI * exp(-params.F / (params.R * params.T) * η_SEI)
    
    # 状态增量
    dC_p_avg = -15 * params.D_p / params.R_p^2 * (C_p_avg - C_p_surf)
    dC_n_avg = -15 * params.D_n / params.R_n^2 * (C_n_avg - C_n_surf)
    dδ_SEI = i_SEI * params.M_SEI / (params.F * params.ρ_SEI * params.a_n * params.l_n)
    dc_f = i_SEI / 3600.0
    
    Δstate = [dC_p_avg * dt, dC_n_avg * dt, dδ_SEI * dt, dc_f * dt]
    
    return Δstate
end

# 批量生成训练数据的函数
function generate_training_data(params::SPMParameters, num_samples=1000; dt=1.0)
    """
    生成用于Kriging模型训练的数据
    返回: (inputs, outputs)
    - inputs: 矩阵，每行是 [state, input]
    - outputs: 矩阵，每行是 Δstate
    """
    inputs = Matrix{Float64}(undef, num_samples, 5)  # 4个状态 + 1个输入
    outputs = Matrix{Float64}(undef, num_samples, 4) # 4个状态增量
    
    for i in 1:num_samples
        # 生成随机状态和输入
        state, input_val = generate_random_state_input(params)
        
        # 计算状态增量
        Δstate = calculate_SPM_state_increment(params, state, input_val, dt)
        
        # 存储数据
        inputs[i, :] = vcat(state, input_val)
        outputs[i, :] = Δstate
        
        if i % 100 == 0
            println("进度: $i/$num_samples")
        end
    end
    
    return inputs, outputs
end

# 保存数据到CSV文件
function save_to_csv(inputs, outputs, params, filename="spm_training_data.csv")
    # 创建DataFrame
    df = DataFrame()
    
    # 输入列
    df.C_p_avg = inputs[:, 1]
    df.C_n_avg = inputs[:, 2]
    df.δ_SEI_nm = inputs[:, 3] .* 1e9  # 转换为nm
    df.c_f_Ah = inputs[:, 4]
    df.P_kW = inputs[:, 5]
    
    # 输出列
    df.ΔC_p_avg = outputs[:, 1]
    df.ΔC_n_avg = outputs[:, 2]
    df.Δδ_SEI_pm = outputs[:, 3] .* 1e12  # 转换为pm
    df.Δc_f_μAh = outputs[:, 4] .* 1e6    # 转换为μAh
    
    # 保存到CSV
    CSV.write(filename, df)
    println("数据已保存到: $filename")
    
    return df
end

# 主函数：生成Kriging训练数据
function main()
    println("SPM训练数据生成开始...")
    
    # 创建SPM参数
    params = create_default_SPM_parameters()
    
    # 生成训练数据
    num_samples = 1000
    println("生成 $num_samples 个训练样本...")
    inputs, outputs = generate_training_data(params, num_samples)
    
    # 保存数据到CSV
    df = save_to_csv(inputs, outputs, params)
    
    # 简要统计
    println("\n数据统计:")
    println("样本数量: $num_samples")
    println("输入维度: 5 [C_p_avg, C_n_avg, δ_SEI, c_f, P]")
    println("输出维度: 4 [ΔC_p_avg, ΔC_n_avg, Δδ_SEI, Δc_f]")
    
    # 显示前几行数据预览
    println("\n数据预览 (前5行):")
    show(first(df, 5))
    println()
    
    return inputs, outputs, params, df
end

# 运行主函数
println("程序启动...")
inputs, outputs, params, df = main()
println("程序执行完成")