import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_cdf(values):
    """Compute sorted values and their corresponding empirical CDF."""
    sorted_vals = np.sort(values)
    cdf = np.arange(1, len(values) + 1) / len(values)
    return sorted_vals, cdf

# 1. Read both CSVs
df_orcaplus = pd.read_csv("/home/jane/Documents/Orca/orca++-uniform.csv")
df_orca = pd.read_csv("/home/jane/Documents/Orca/orca-uniform.csv")

# 2. Merge on trace_id
#    The suffixes make it clear which columns came from orcaplus vs. orca.
df_merged = df_orcaplus.merge(df_orca, on="trace_id", suffixes=("_plus", "_orca"))

# 3. Compute metrics of interest
#    Utilization difference (improvement): orcaplus.util% - orca.util%
df_merged["util_improvement"] = df_merged["util%_plus"] - df_merged["util%_orca"]

#    Delay ratio: orcaplus.davg / orca.davg
df_merged["delay_ratio"] = df_merged["davg_plus"] / df_merged["davg_orca"]

# 4. Compute the CDFs
util_improv_sorted, util_improv_cdf = compute_cdf(df_merged["util_improvement"])
delay_ratio_sorted, delay_ratio_cdf = compute_cdf(df_merged["delay_ratio"])

# 5. Plot the utilization improvement CDF
plt.figure(figsize=(8, 4))
plt.plot(util_improv_sorted, util_improv_cdf, label="Util Improvement")
plt.xlabel("Orca++ Utilization - Orca Utilization", fontsize=16)
plt.ylabel("CDF", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.legend(loc="lower right", fontsize=14, frameon=False)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("orca_util_improvement_CDF.png", dpi=300)
plt.show()

# 6. Plot the average delay ratio CDF
plt.figure(figsize=(8, 4))
plt.plot(delay_ratio_sorted, delay_ratio_cdf, label="Delay Ratio (Orca+ / Orca)")
plt.xlabel("Orca++ Delay / Orca Delay", fontsize=16)
plt.ylabel("CDF", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.legend(loc="lower right", fontsize=14, frameon=False)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("orca_delay_ratio_CDF.png", dpi=300)
plt.show()
