import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.decomposition import PCA
from tqdm import tqdm
import imageio

# ==== CONFIG ====
video_path = "C:/Users/l/Downloads/waveclip.mp4"
output_dir = "C:/Users/l/Downloads/smallscale_wave_analysis"
os.makedirs(output_dir, exist_ok=True)
fps = 30
k_values = [1, 5, 10, 20, 50, 100]

# ==== HELPER FUNCTIONS ====
def read_frames(video_path):

    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    return frames

def flatten_frame(img, gamma=1.8):
    """Flatten frame with gamma correction"""
    # Apply gamma correction
    img_float = img.astype(np.float32) / 255.0
    img_gamma = np.power(img_float, gamma) * 255
    img_gamma = np.clip(img_gamma, 0, 255).astype(np.uint8)
    
    # Flatten RGB channels
    r = img_gamma[:, :, 0].flatten()
    g = img_gamma[:, :, 1].flatten()
    b = img_gamma[:, :, 2].flatten()
    
    return np.concatenate([r, g, b])

def reconstruct_frame(vec, height, width, gamma=1.8):
    """Reconstruct frame with inverse gamma correction"""
    n_pix = height * width
    
    # Extract RGB channels
    r_vec = vec[:n_pix]
    g_vec = vec[n_pix:2*n_pix]
    b_vec = vec[2*n_pix:3*n_pix]
    
    # Reverse gamma correction
    r_vec = np.clip(r_vec, 0, 255).astype(np.float32) / 255.0
    g_vec = np.clip(g_vec, 0, 255).astype(np.float32) / 255.0
    b_vec = np.clip(b_vec, 0, 255).astype(np.float32) / 255.0
    
    r_vec = np.power(r_vec, 1/gamma) * 255
    g_vec = np.power(g_vec, 1/gamma) * 255
    b_vec = np.power(b_vec, 1/gamma) * 255
    
    # Reshape to image dimensions
    r_mat = np.clip(r_vec.reshape(height, width), 0, 255).astype(np.uint8)
    g_mat = np.clip(g_vec.reshape(height, width), 0, 255).astype(np.uint8)
    b_mat = np.clip(b_vec.reshape(height, width), 0, 255).astype(np.uint8)
    
    # Combine channels
    reconstructed = np.stack([r_mat, g_mat, b_mat], axis=-1)
    return reconstructed

# ==== MAIN PROCESSING ====
print("Reading frames from video...")
frames = read_frames(video_path)
n_frames = len(frames)
height, width = frames[0].shape[:2]

print(f"Loaded {n_frames} frames of size {width}x{height}")


# Flatten all frames
print("Flattening all frames for PCA...")
data_mat = np.array([flatten_frame(frame) for frame in tqdm(frames)])
print(f"Data matrix dimensions: {data_mat.shape}")

# Perform PCA
print("Performing PCA...")
pca = PCA()
pca_scores = pca.fit_transform(data_mat)
var_explained = pca.explained_variance_
total_var = np.sum(var_explained)
cumvar = np.cumsum(var_explained) / total_var * 100
individual_var = var_explained / total_var * 100

print("\n=== VARIANCE EXPLAINED BY COMPONENTS ===")
for k in range(1, min(11, len(individual_var))):
    print(f"k={k:2d}: Individual = {individual_var[k-1]:6.2f}%, Cumulative = {cumvar[k-1]:6.2f}%")
print()

# ==== VIDEO RECONSTRUCTION FUNCTION ====
def reconstruct_video(k):
    print(f"=== RECONSTRUCTING WITH k={k} COMPONENTS ===")
    
    k_actual = min(k, pca.n_components_)
    if k_actual != k:
        print(f"Warning: Reduced k from {k} to {k_actual} (max available)")
    
    output_dir_k = os.path.join(output_dir, f"frames_k{k_actual:03d}")
    os.makedirs(output_dir_k, exist_ok=True)
    
    # Reconstruct and save frames
    print(f"Reconstructing {n_frames} frames...")
    
    for i in tqdm(range(n_frames)):
        scores = pca_scores[i, :k_actual]
        loadings = pca.components_[:k_actual, :]
        recon_vec = scores @ loadings + pca.mean_
        
        frame_img = reconstruct_frame(recon_vec, height, width)
        output_path = os.path.join(output_dir_k, f"frame_{i:04d}.png")
        imageio.imwrite(output_path, frame_img)
    var_pct = cumvar[k_actual - 1] if k_actual <= len(cumvar) else 100
    
    print(f" Frames saved in: {output_dir_k}")
    print(f" Variance explained: {var_pct:.2f}%")
    print()
    
    return {"k": k_actual, "variance_explained": var_pct}

# ==== RUN ALL RECONSTRUCTIONS ====
print("Starting video reconstructions...")
reconstruction_results = []

for k in k_values:
    if k <= min(data_mat.shape[0], data_mat.shape[1]):
        results = reconstruct_video(k)
        reconstruction_results.append(results)
    else:
        print(f"Skipping k={k} (exceeds {min(data_mat.shape[0], data_mat.shape[1])} available components)\n")

print("=" * 40)
print("              SUMMARY")
print("=" * 40)
print(f"Video: {os.path.basename(video_path)}")
print(f"Frames: {n_frames} | Dimensions: {width}x{height}")
print()

print("RECONSTRUCTIONS COMPLETED:")
for result in reconstruction_results:
    print(f"  k={result['k']:3d}: {result['variance_explained']:6.2f}% variance explained")



# === TOP 3 SPATIAL MODES ===
print("Extracting top 3 spatial modes...")

def normalize_to_255(img):
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-6:
        img_norm = (img - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(img)
    return (img_norm * 255).astype(np.uint8)

top_n = 3
spatial_modes = []

for i in range(top_n):
    mode_vec = pca.components_[i]
    
    n_pix = height * width
    r_mode = mode_vec[:n_pix].reshape(height, width)
    g_mode = mode_vec[n_pix:2*n_pix].reshape(height, width)
    b_mode = mode_vec[2*n_pix:3*n_pix].reshape(height, width)
    
    gray_mode = 0.2989 * r_mode + 0.5870 * g_mode + 0.1140 * b_mode  
    gray_mode = normalize_to_255(gray_mode)
    
    spatial_modes.append(gray_mode)

fig, axes = plt.subplots(1, top_n, figsize=(15, 5))
for i in range(top_n):
    axes[i].imshow(spatial_modes[i], cmap="gray")
    axes[i].set_title(f"Spatial Mode {i+1}", fontsize=12)
    axes[i].axis('off')

plt.suptitle("Top 3 Spatial Modes", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.8)  

plt.savefig(os.path.join(output_dir, "top3_spatial_modes.png"), dpi=300)
plt.show()
plt.close()

print(" Saved top 3 spatial modes side-by-side in output directory.")



# ==== SEISMIC-STYLE TEMPORAL COMPONENTS PLOT ====
print("Creating seismic-style temporal components plot...")

num_modes_to_plot = 10  
base_sep = 2.0            
band_frac = 0.8        

time_axis = np.arange(n_frames) / fps

temporal_modes = pca_scores[:, :num_modes_to_plot]

eigenvalues = pca.explained_variance_[:num_modes_to_plot]

scaled_modes = temporal_modes * np.sqrt(eigenvalues)[np.newaxis, :]

peak_to_peak = scaled_modes.max(axis=0) - scaled_modes.min(axis=0)
scaled_modes = (scaled_modes / (peak_to_peak + 1e-12)) * (band_frac * base_sep)

offsets = np.arange(num_modes_to_plot, 0, -1) * base_sep
stacked_modes = scaled_modes + offsets[np.newaxis, :]

fig, ax = plt.subplots(figsize=(18, 12))

colors = plt.cm.plasma(np.linspace(0.1, 0.9, num_modes_to_plot))

for i in range(num_modes_to_plot):
    ax.plot(time_axis, stacked_modes[:, i], 
            linewidth=0.8, color=colors[i], alpha=0.9)
    
    ax.axhline(offsets[i], linestyle=":", linewidth=0.8, 
               color="black", alpha=0.7)
    
    variance_pct = individual_var[i] if i < len(individual_var) else 0
    ax.text(time_axis[0] - 0.5, offsets[i], 
            f"PC{i+1}", 
            va="center", ha="right", fontsize=10, weight="bold")

ax.set_xlim(time_axis[0], time_axis[-1])
ax.set_xlabel("Time (seconds)", fontsize=12)
ax.set_title(f"Seismic-style PCA Temporal Components (Top {num_modes_to_plot})", 
             fontsize=16, weight="bold")

ax.yaxis.set_visible(False)
for spine in ("left", "right", "top"):
    ax.spines[spine].set_visible(False)

ax.grid(True, axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"temporal_components_seismic_top{num_modes_to_plot}.png"), 
            dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print(f"Saved seismic-style temporal PCA components plot (first {num_modes_to_plot})")




print("Creating variance explained plot...")

k_values = [1, 5, 10, 20, 50, 100]
video_pca_variance = [10.54, 29.32, 39.16, 51.04, 66.09, 76.27]

plt.figure(figsize=(12, 8))

plt.plot(k_values, video_pca_variance, 'o-', linewidth=3, markersize=8,
         label='Video PCA (Spatial-Temporal)', color='#2E86AB', alpha=0.8)

plt.annotate(f'{video_pca_variance[0]:.1f}% with k=1', 
             xy=(1, video_pca_variance[0]), xytext=(20, 25),
             arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
             fontsize=11, color='gray', fontweight='bold')

plt.annotate(f'{video_pca_variance[-1]:.1f}% at k=100', 
             xy=(100, video_pca_variance[-1]), xytext=(60, 85),
             arrowprops=dict(arrowstyle='->', color='darkgreen', alpha=0.7),
             fontsize=11, color='darkgreen', fontweight='bold')

plt.xlabel('Number of Components (k)', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative Variance Explained (%)', fontsize=14, fontweight='bold')
plt.title('Video PCA: Cumulative Variance Explained\n(Spatial-Temporal Analysis)',
          fontsize=16, fontweight='bold', pad=20)

plt.xscale('log')
plt.xlim(0.8, 120)
plt.ylim(0, 85)

plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12, loc='center right', frameon=True, fancybox=True, shadow=True)

textstr = f'First component captures {video_pca_variance[0]:.1f}% of total variance\nTop 100 components capture {video_pca_variance[-1]:.1f}%'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

plt.tight_layout()

plt.savefig(os.path.join(output_dir, "vpca_variance_explained.png"),
            dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print(" PCA variance explained plot saved in output directory)")



print("CREATING  COMPARISON GRID...")

compare_k_values = [5, 10, 20, 50, 100]
compare_times = [n_frames // 4, n_frames // 2, 3 * n_frames // 4]

fig, axs = plt.subplots(len(compare_times), len(compare_k_values) + 1, 
                        figsize=(20, 8),
                        gridspec_kw={'wspace': 0.05, 'hspace': 0.1})

if len(compare_times) == 1:
    axs = axs.reshape(1, -1)

for col, k in enumerate(compare_k_values):
    axs[0, col + 1].set_title(f"k={k}", fontsize=12, pad=8)

for row, t in enumerate(compare_times):
    time_sec = t / fps
    mins = int(time_sec // 60)
    secs = int(time_sec % 60)
    axs[row, 0].imshow(frames[t])
    axs[row, 0].set_title(f"Original {mins:02d}:{secs:02d}", fontsize=10, pad=5)
    axs[row, 0].axis("off")

    for col, k in enumerate(compare_k_values):
        k_actual = min(k, pca.n_components_)
        frame_path = os.path.join(output_dir, f"frames_k{k_actual:03d}", f"frame_{t:04d}.png")
        if os.path.exists(frame_path):
            recon_img = imageio.imread(frame_path)
            axs[row, col + 1].imshow(recon_img)
        else:
            axs[row, col + 1].text(0.5, 0.5, f"k={k}\nnot found", 
                                   ha='center', va='center', transform=axs[row, col + 1].transAxes,
                                   fontsize=8)
        axs[row, col + 1].axis("off")

plt.suptitle("Original vs. Reconstructed Frames at Selected Times and k-values", 
             fontsize=14, y=0.95)
plt.tight_layout()
plt.subplots_adjust(top=0.88)  
plt.savefig(os.path.join(output_dir, "comparison_grid.png"), dpi=150, bbox_inches='tight')
plt.show()

print(" comparison grid saved in output directory")
