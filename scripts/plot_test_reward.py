import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def extract_rewards_and_times(file_path):
    rewards = []
    times = []
    with open(file_path, 'r') as f:
        for line in f:
            time_match = re.search(r"Time:\s*([\d.]+)", line)
            reward_match = re.search(r"raw_reward:\s*([0-9.]+)", line)
            if time_match and reward_match:
                timestamp = float(time_match.group(1))
                reward = float(reward_match.group(1))
                times.append(timestamp)
                rewards.append(reward)
    return times, rewards

def collect_aligned_rewards(base_dir, label):
    run_results = defaultdict(dict)  # {trace_name: {label-runX: (timestamps, rewards)}}
    for run in ["run0", "run1"]:
        run_dir = os.path.join(base_dir, run, "evaluation_log")
        if not os.path.exists(run_dir):
            continue
        for fname in os.listdir(run_dir):
            if not fname.endswith("reward_and_state.txt"):
                continue
            match = re.search(r"xtwo1-(.*?)-1-25-reward_and_state\.txt", fname)
            if not match:
                continue
            trace_name = match.group(1)
            full_path = os.path.join(run_dir, fname)
            times, rewards = extract_rewards_and_times(full_path)
            if times and rewards:
                run_results[trace_name][f"{label}-{run}"] = (times, rewards)
    return run_results

def avg_over_runtime(times, rewards, runtime_min):
    start_time = times[0]
    end_time = start_time + runtime_min
    filtered = [r for t, r in zip(times, rewards) if start_time <= t <= end_time]
    return np.mean(filtered) if filtered else np.nan

# Set base paths
orca_base = os.path.expanduser("~/Desktop/orca-results")
orcapp_base = os.path.expanduser("~/Desktop/orca++-results")

# Collect reward-time data
orca_time_data = collect_aligned_rewards(orca_base, "orca")
orcapp_time_data = collect_aligned_rewards(orcapp_base, "orca++")

# Compare using smallest shared runtime across all logs
orca_vals, orcapp_vals = [], []

for trace in sorted(set(orca_time_data.keys()) & set(orcapp_time_data.keys())):
    entries = {
        **orca_time_data[trace],
        **orcapp_time_data[trace]
    }
    keys_needed = ["orca-run0", "orca-run1", "orca++-run0", "orca++-run1"]
    if all(k in entries for k in keys_needed):
        try:
            # Determine the minimum runtime across all 4 logs
            runtimes = [times[-1] - times[0] for (times, _) in entries.values()]
            runtime_min = min(runtimes)
            print(f"Trace: {trace}, min runtime: {runtime_min:.2f} seconds")

            orca_avg = np.mean([
                avg_over_runtime(*entries["orca-run0"], runtime_min),
                avg_over_runtime(*entries["orca-run1"], runtime_min),
            ])

            orcapp_avg = np.mean([
                avg_over_runtime(*entries["orca++-run0"], runtime_min),
                avg_over_runtime(*entries["orca++-run1"], runtime_min),
            ])

            if not (np.isnan(orca_avg) or np.isnan(orcapp_avg)):
                orca_vals.append(orca_avg)
                orcapp_vals.append(orcapp_avg)
        except Exception as e:
            print(f"Skipping trace {trace} due to error: {e}")

# Plot CDF of reward ratios
ratios = np.array(orcapp_vals) / np.array(orca_vals)
sorted_ratios = np.sort(ratios)
cdf = np.arange(len(sorted_ratios)) / len(sorted_ratios)

plt.figure()
plt.plot(sorted_ratios, cdf, label="CDF of Orca++ / Orca Test Reward (Min Runtime)")
plt.xlabel("Orca++ Test Reward / Orca Test Reward")
plt.ylabel("CDF")
plt.title("CDF of Test Reward Ratio Over Minimum Runtime (Orca++ vs. Orca)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("orca_vs_orcapp_cdf.png")
plt.show()
