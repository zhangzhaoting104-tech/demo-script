# test_corrected.jl
println("="^60)
println("测试修正后的Kriging模型")
println("="^60)

# 加载模块
include("kriging_module.jl")

# 训练模型
model, predictor = KrigingModelModule.create_and_train_kriging_model("spm_training_data.csv", verbose=true)

# 保存模型（可选）
println("\n" * "="^60)
println("测试完成！")
println("="^60)