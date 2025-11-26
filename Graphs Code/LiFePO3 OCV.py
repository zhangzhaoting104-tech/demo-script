import numpy as np
import matplotlib.pyplot as plt

def OCV_positive(theta_p):
    if theta_p < 0.05:
        return 3.30 + 2.0 * theta_p
    elif theta_p < 0.95:  
        return 3.40  # 主要平坦平台
    else:
        # 当theta_p >= 0.95时的返回值
        return 3.40 + 2.0 * (theta_p - 0.95)  # 示例：添加一个斜率

def OCV_negative(theta_n):
    return 0.12 + 0.15 * theta_n + 0.05 * theta_n**2

# Generate SOC data
soc = np.linspace(0, 1, 100)

# Calculate OCV values
ocv_p = [OCV_positive(x) for x in soc]  # 使用列表推导式处理每个值
ocv_n = OCV_negative(soc)

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(soc, ocv_p, 'b-', linewidth=2)
plt.xlabel('SOC')
plt.ylabel('Voltage (V)')
plt.title('Positive Electrode OCV (LiFePO₄)')
plt.grid(True, alpha=0.3)
plt.ylim(3.0, 4.0)

plt.subplot(1, 2, 2)
plt.plot(soc, ocv_n, 'r-', linewidth=2)
plt.xlabel('SOC')
plt.ylabel('Voltage (V)')
plt.title('Negative Electrode OCV (Graphite)')
plt.grid(True, alpha=0.3)
plt.ylim(0.0, 0.7)

plt.tight_layout()
plt.show()