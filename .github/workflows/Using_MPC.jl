using .BSSMPCFramework

# 运行演示
controller, results = BSSMPCFramework.run_mpc_demo(
    K=3,           # 3个电池
    N=12,          # 12小时预测
    total_hours=48 # 48小时仿真
)
# 逐步运行，24小时预测，72小时仿真
# controller = BSSMPCFramework.create_test_scenario(K=21, N=24, total_hours=72)
# controller = BSSMPCFramework.run_mpc_rolling(controller)
# results = BSSMPCFramework.analyze_mpc_results(controller)
# BSSMPCFramework.visualize_power_decisions(controller, 10)