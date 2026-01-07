import faiss
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# --- 1. SYSTEM CONFIGURATION & DATA GENERATION ---
np.random.seed(42)
d = 128          # Original dimension
nb = 100000      # Database size
nq = 1000        # Number of queries
k = 10           # Top-K

print(f"Generating structured manifold data (d={d}, {nb} vectors)...")
xb, _ = make_blobs(n_samples=nb, n_features=d, centers=50, cluster_std=2.0, random_state=42)
xq, _ = make_blobs(n_samples=nq, n_features=d, centers=50, cluster_std=2.0, random_state=42)
xb = xb.astype('float32')
xq = xq.astype('float32')

# --- 2. FACTOR ANALYSIS LAYER (PCA for decorrelation) ---
print("Applying Factor Analysis via PCA (reducing to 64 dimensions + whitening)...")
pca_dim = 64
pca = PCA(n_components=pca_dim, whiten=True, random_state=42)
xb_optimized = pca.fit_transform(xb).astype('float32')
xq_optimized = pca.transform(xq).astype('float32')

# --- 3. HYBRID CLUSTERED INDEX (IVF on optimized space) ---
nlist = 128
print("Building Hybrid Factor-Clustered IVF Index...")
quantizer = faiss.IndexFlatL2(pca_dim)
index = faiss.IndexIVFFlat(quantizer, pca_dim, nlist, faiss.METRIC_L2)
index.train(xb_optimized)
index.add(xb_optimized)

# Ground truth on the optimized space
print("Computing exact ground truth...")
index_gt = faiss.IndexFlatL2(pca_dim)
index_gt.add(xb_optimized)
_, gt = index_gt.search(xq_optimized, k)

# --- 4. DEA-INSPIRED EFFICIENCY SCORING ---
def calculate_dea_efficiency(recall, qps):
    """Simplified DEA: Output (recall) / Input (inverse throughput). Use log for scale."""
    if qps <= 0:
        return 0.0
    return recall * np.log10(qps + 1e-8)  # small epsilon for safety

def evaluate_system(index, xq_query, nprobe_list):
    stats = []
    for n in nprobe_list:
        index.nprobe = n
        start = time.time()
        _, I = index.search(xq_query, k)
        elapsed = time.time() - start
        qps = nq / elapsed if elapsed > 0 else 0.001
        
        # Fast vectorized recall
        recall = np.mean([
            len(np.intersect1d(I[i], gt[i], assume_unique=True)) / k 
            for i in range(nq)
        ])
        
        efficiency = calculate_dea_efficiency(recall, qps)
        
        stats.append({
            'nprobe': n,
            'recall': recall,
            'qps': qps,
            'efficiency': efficiency
        })
        print(f"  nprobe={n:2d} → Recall={recall:.3f}, QPS={qps:.1f}, Efficiency={efficiency:.4f}")
    
    return stats

# --- 5. BENCHMARK & VISUALIZATION ---
nprobes = [1, 2, 4, 8, 16, 32, 64]
print("\nEvaluating system performance...")
results = evaluate_system(index, xq_optimized, nprobes)

recalls = [r['recall'] for r in results]
qps_vals = [r['qps'] for r in results]
eff_scores = [r['efficiency'] for r in results]
n_labels = [r['nprobe'] for r in results]

# Plot
fig, ax1 = plt.subplots(figsize=(12, 8))

# Recall vs QPS line
ax1.set_xlabel('Queries Per Second (Higher = Better)', fontsize=13)
ax1.set_ylabel('Recall@10 (Higher = Better)', color='tab:blue', fontsize=13)
ax1.plot(qps_vals, recalls, 'o-', color='tab:blue', linewidth=3, markersize=8, label='Pareto Frontier')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, alpha=0.3)

# DEA Efficiency bars
ax2 = ax1.twinx()
ax2.set_ylabel('DEA-Inspired Efficiency Score\n(Recall × log₁₀(QPS))', color='tab:red', fontsize=13)
bar_width = (max(qps_vals) - min(qps_vals)) / len(qps_vals) * 0.6
ax2.bar(qps_vals, eff_scores, width=bar_width, alpha=0.35, color='tab:red', label='Efficiency Score')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Annotations
for i, txt in enumerate(n_labels):
    ax1.annotate(f"nprobe={txt}", (qps_vals[i], recalls[i]), 
                 xytext=(0, 10), textcoords='offset points', fontsize=10, ha='center')

plt.title('Efficient Frontier Analysis of Hybrid Factor-Clustered Vector Search\n'
          'PCA Preprocessing + IVF Clustering (Relevant to Advanced Vector DB Optimization)',
          fontsize=15, pad=20)
fig.tight_layout()
plt.show()

# --- 6. OPTIMAL CONFIGURATION ---
best_idx = int(np.argmax(eff_scores))
best_config = results[best_idx]
print(f"\n{'='*50}")
print(f"RESEARCH RESULT: MATHEMATICALLY OPTIMAL CONFIGURATION")
print(f"{'='*50}")
print(f"nprobe             : {best_config['nprobe']}")
print(f"Recall@10          : {best_config['recall']:.2%}")
print(f"Throughput         : {best_config['qps']:.1f} queries/second")
print(f"DEA Efficiency Score: {best_config['efficiency']:.4f} (maximum)")
print(f"{'='*50}")
