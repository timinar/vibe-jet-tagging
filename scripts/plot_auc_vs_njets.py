"""
Plot AUC vs number of jets to investigate dependency on sample size.
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the result files
data = {
    100: {
        "low": {"auc_mean": 0.8704043560120498, "auc_ci_low": 0.8081577493686869, "auc_ci_high": 0.929828947368421},
        "medium": {"auc_mean": 0.8687918906911097, "auc_ci_low": 0.8030129190455279, "auc_ci_high": 0.9259259259259259}
    },
    300: {
        "low": {"auc_mean": 0.7784624062363402, "auc_ci_low": 0.7318108974358974, "auc_ci_high": 0.8261991713844772},
        "medium": {"auc_mean": 0.7784828952807383, "auc_ci_low": 0.7291626738277767, "auc_ci_high": 0.8248081014071004}
    },
    500: {
        "low": {"auc_mean": 0.7810874034843479, "auc_ci_low": 0.7435582756173406, "auc_ci_high": 0.8168430294072},
        "medium": {"auc_mean": 0.7826385763888847, "auc_ci_low": 0.7472069499034816, "auc_ci_high": 0.8172742720459281}
    },
    1000: {
        "low": {"auc_mean": 0.7527901896700737, "auc_ci_low": 0.7269136341881397, "auc_ci_high": 0.7802022518786373},
        "medium": {"auc_mean": 0.7530370468443707, "auc_ci_low": 0.7272989701315371, "auc_ci_high": 0.7772710368774679}
    },
    3000: {
        "low": {"auc_mean": 0.7676136829147926, "auc_ci_low": 0.7521616951638478, "auc_ci_high": 0.7816152551736767},
        "medium": {"auc_mean": 0.7675842043090334, "auc_ci_low": 0.75301364500198, "auc_ci_high": 0.7823959026759277}
    }
}

# Extract data for plotting
n_jets = sorted(data.keys())
reasoning_efforts = ["low", "medium"]

fig, ax = plt.subplots(figsize=(10, 6))

colors = {"low": "blue", "medium": "red"}
markers = {"low": "o", "medium": "s"}

for effort in reasoning_efforts:
    auc_means = [data[n][effort]["auc_mean"] for n in n_jets]
    auc_ci_low = [data[n][effort]["auc_ci_low"] for n in n_jets]
    auc_ci_high = [data[n][effort]["auc_ci_high"] for n in n_jets]

    # Calculate error bars
    err_low = [auc_means[i] - auc_ci_low[i] for i in range(len(n_jets))]
    err_high = [auc_ci_high[i] - auc_means[i] for i in range(len(n_jets))]

    ax.errorbar(n_jets, auc_means,
                yerr=[err_low, err_high],
                fmt=markers[effort] + '-',
                color=colors[effort],
                label=f"{effort} reasoning",
                capsize=5,
                markersize=8,
                linewidth=2,
                capthick=2)

ax.set_xlabel("Number of Jets", fontsize=14)
ax.set_ylabel("AUC (Bootstrap Mean ± 95% CI)", fontsize=14)
ax.set_title("AUC vs Number of Jets\n(with_optimal_cut template)", fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Add more x-axis tick labels
ax.set_xticks(n_jets)
ax.set_xticklabels(n_jets)
ax.tick_params(labelsize=12)

# Highlight the problematic behavior
ax.axhline(y=0.75, color='gray', linestyle='--', alpha=0.5, label='AUC = 0.75')

plt.tight_layout()
plt.savefig("results/auc_vs_njets.png", dpi=300, bbox_inches='tight')
plt.savefig("results/auc_vs_njets.pdf", bbox_inches='tight')
print("Plots saved to results/auc_vs_njets.png and results/auc_vs_njets.pdf")

# Print summary statistics
print("\n" + "="*80)
print("AUC Summary Statistics")
print("="*80)
print(f"{'N Jets':<10} {'Reasoning':<10} {'AUC Mean':<12} {'CI Width':<12} {'95% CI'}")
print("-"*80)
for n in n_jets:
    for effort in reasoning_efforts:
        d = data[n][effort]
        ci_width = d["auc_ci_high"] - d["auc_ci_low"]
        print(f"{n:<10} {effort:<10} {d['auc_mean']:<12.4f} {ci_width:<12.4f} "
              f"[{d['auc_ci_low']:.4f}, {d['auc_ci_high']:.4f}]")

# Statistical analysis
print("\n" + "="*80)
print("Analysis of CI Overlap")
print("="*80)

def check_overlap(n1, n2, effort):
    """Check if confidence intervals overlap between two sample sizes"""
    ci1_low = data[n1][effort]["auc_ci_low"]
    ci1_high = data[n1][effort]["auc_ci_high"]
    ci2_low = data[n2][effort]["auc_ci_low"]
    ci2_high = data[n2][effort]["auc_ci_high"]

    # Check for any overlap
    overlap = not (ci1_high < ci2_low or ci2_high < ci1_low)
    return overlap

# Check overlap between 100 and 3000 jets
for effort in reasoning_efforts:
    overlap = check_overlap(100, 3000, effort)
    print(f"\n{effort.capitalize()} reasoning:")
    print(f"  100 jets:  AUC = {data[100][effort]['auc_mean']:.4f}, "
          f"CI = [{data[100][effort]['auc_ci_low']:.4f}, {data[100][effort]['auc_ci_high']:.4f}]")
    print(f"  3000 jets: AUC = {data[3000][effort]['auc_mean']:.4f}, "
          f"CI = [{data[3000][effort]['auc_ci_low']:.4f}, {data[3000][effort]['auc_ci_high']:.4f}]")
    print(f"  CI overlap: {overlap}")
    print(f"  Mean difference: {data[100][effort]['auc_mean'] - data[3000][effort]['auc_mean']:.4f}")

# Calculate relative CI widths
print("\n" + "="*80)
print("Relative CI Width (normalized by √N)")
print("="*80)
print(f"{'N Jets':<10} {'Reasoning':<10} {'CI Width':<12} {'CI Width × √N':<15} {'Relative'}")
print("-"*80)
reference_n = 100
for n in n_jets:
    for effort in reasoning_efforts:
        d = data[n][effort]
        ci_width = d["auc_ci_high"] - d["auc_ci_low"]
        normalized = ci_width * np.sqrt(n)
        relative = normalized / (data[reference_n][effort]["auc_ci_high"] - data[reference_n][effort]["auc_ci_low"])
        relative /= np.sqrt(reference_n)
        print(f"{n:<10} {effort:<10} {ci_width:<12.4f} {normalized:<15.2f} {relative:<10.2f}")
