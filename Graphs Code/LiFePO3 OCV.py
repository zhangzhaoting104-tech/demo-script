import numpy as np
import matplotlib.pyplot as plt

def OCV_positive(theta_p):
    return 3.4 + 0.5 * (1 - np.exp(-5 * (1 - theta_p))) - 0.3 * (1 - np.exp(-5 * theta_p))

def OCV_negative(theta_n):
    return 0.1 + 0.8 * theta_n - 0.3 * theta_n**2

# Generate SOC data
soc = np.linspace(0, 1, 100)

# Calculate OCV values
ocv_p = OCV_positive(soc)
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