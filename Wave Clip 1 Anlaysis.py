import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#CONFIG 
video_path = os.path.expanduser("~/Downloads/wave1_clip_final_1080p.mp4")
output_dir = os.path.expanduser("~/Downloads/wave_output")
os.makedirs(output_dir, exist_ok=True)
resize_dim = (256,144)

# HELPERS
def get_video_fps(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def read_video(path, max_frames=None, resize_dim=None):
    cap = cv2.VideoCapture(path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames and count >= max_frames):
            break
        if resize_dim is not None:
            frame = cv2.resize(frame, resize_dim)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)
        count += 1
    cap.release()
    return np.array(frames)

def save_video(frames_rgb, output_path, fps):
    h, w, _ = frames_rgb[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames_rgb:
        bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(bgr)
    out.release()

def save_image(image, path, title="", cmap='gray'):
    plt.figure()
    plt.imshow(image.astype(np.uint8), cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.savefig(path)
    plt.close()

# Read and resize video 
video = read_video(video_path, resize_dim=resize_dim)
T, H, W, C = video.shape
print("Loaded video:", video.shape)

#Compute dominant color basis (from whole video) 
video_flat = video.reshape(-1, 3)
mean_rgb = video_flat.mean(axis=0)
std_rgb = video_flat.std(axis=0)
normalized = (video_flat - mean_rgb) / std_rgb
corr = np.corrcoef(normalized.T)
eigvals, eigvecs = np.linalg.eig(corr)
top3_vecs = eigvecs[:, np.argsort(eigvals)[-3:]]  
dominant_vec = top3_vecs[:, -1]  

# Project entire video onto dominant direction 
projected = np.empty((T, H, W))
for t in range(T):
    frame_flat = video[t].reshape(-1, 3)
    projection = (frame_flat - mean_rgb) @ dominant_vec  
    projected[t] = projection.reshape(H, W)

# Normalize (standardize) projected video ===
mean_image = projected.mean(axis=0)
std_image = projected.std(axis=0)
standardized = (projected - mean_image) / std_image


# Normalize mean_image and std_image 
def normalize_to_255(img):
    img_min, img_max = np.min(img), np.max(img)
    norm_img = (img - img_min) / (img_max - img_min) * 255
    return norm_img

mean_image_norm = normalize_to_255(mean_image)
std_image_norm = normalize_to_255(std_image)

save_image(mean_image_norm, os.path.join(output_dir, "mean_projected.png"), "Mean Projected")
save_image(std_image_norm, os.path.join(output_dir, "std_projected.png"), "Std Dev Projected")

#  Save y matrix as (pixels, time) 

y = standardized.reshape(T, H*W).T  # shape (pixels, time)
np.savetxt(os.path.join(output_dir, "y_data_2.csv"), y, delimiter=",")
print(f"Saved y_data.csv with shape {y.shape} to {output_dir}")

#  Apply PCA on standardized projected video (spatial PCA over time) ===
X = standardized.reshape(T, -1)
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X)
print("Explained variance ratio (top 5):", pca.explained_variance_ratio_[:5])


#  full PCA to 200 
pca_full = PCA(n_components=200)
X_full_pca = pca_full.fit_transform(X)


# === Reconstruct for various k-values ===
fps = get_video_fps(video_path)
k_values = [1, 5, 10, 20,50,100,200]
all_recons = {}

for k in k_values:
    pca_k = PCA(n_components=k)
    X_k = pca_k.fit_transform(X)
    Xk_reconstructed = pca_k.inverse_transform(X_k)
    reconstructed = Xk_reconstructed.reshape(T, H, W) * std_image + mean_image

    # Back-project to RGB using dominant direction + mean RGB
    rgb_recons = []
    for t in range(T):
        flat = reconstructed[t].flatten()
        rgb_flat = (flat[:, None] * dominant_vec[None, :]) + mean_rgb  
        rgb_frame = rgb_flat.reshape(H, W, 3)
        rgb_frame = np.clip(rgb_frame, 0, 255)
        rgb_recons.append(rgb_frame.astype(np.uint8))
    all_recons[k] = rgb_recons

    output_path = os.path.join(output_dir, f"reconstructed_k{k}.mp4")
    save_video(rgb_recons, output_path, fps)
    print(f"Saved reconstructed video with k={k} to:", output_path)


#Variance on V-Data
V = standardized.reshape(T, -1)  # shape: (T, H*W)

# Perform PCA on V-domain data
pca_v = PCA(n_components=min(T, H*W))  
X_v = pca_v.fit_transform(V)

# Cumulative variance explained
cum_var_v = np.cumsum(pca_v.explained_variance_ratio_) * 100

print("V-domain PCA variance explained (first 10 components):")
for i, var in enumerate(pca_v.explained_variance_ratio_[:10] * 100, 1):
    print(f"PC{i}: {var:.2f}%")

print(f"\nCumulative variance explained (first 10 PCs): {cum_var_v[9]:.2f}%")

k_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
k_values = [k for k in k_values if k <= len(cum_var_v)]


print("\nV-domain cumulative variance at selected k-values:")
for k in k_values:
    print(f"V-domain PCA cumulative variance at k={k}: {cum_var_v[k-1]:.2f}%")


#Temporal Correlation Variance Explained
V_raw = projected.reshape(T, -1) 

temporal_corr = np.corrcoef(V_raw)  # shape (T, T)

eigvals, eigvecs = np.linalg.eigh(temporal_corr)

eigvals = eigvals[::-1]
cum_var_temporal = np.cumsum(eigvals) / np.sum(eigvals) * 100

k_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
k_values = [k for k in k_values if k <= len(cum_var_temporal)]
for k in k_values:
    print(f"Temporal correlation PCA cumulative variance at k={k}: {cum_var_temporal[k-1]:.2f}%")
  

# === COMBINED VARIANCE EXPLAINED PLOT ===
import matplotlib.pyplot as plt
import numpy as np

k_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
temporal_variance = [51.41, 52.00, 52.61, 53.66, 56.26, 59.60, 64.56, 73.91, 82.74]
v_domain_variance = [0.59, 1.91, 3.15, 5.35, 10.70, 17.59, 27.77, 46.84, 64.83]

plt.figure(figsize=(12, 8))

plt.plot(k_values, temporal_variance, 'o-', linewidth=3, markersize=8, 
         label='Temporal Correlation PCA', color='#2E86AB', alpha=0.8)
plt.plot(k_values, v_domain_variance, 's-', linewidth=3, markersize=8, 
         label='V-Domain PCA', color='#A23B72', alpha=0.8)

plt.annotate('51.41% at k=1', xy=(1, 51.41), xytext=(20, 65),
            arrowprops=dict(arrowstyle='->', color='#2E86AB', alpha=0.7),
            fontsize=11, color='#2E86AB', fontweight='bold')

plt.annotate('Only 0.59% at k=1', xy=(1, 0.59), xytext=(50, 15),
            arrowprops=dict(arrowstyle='->', color='#A23B72', alpha=0.7),
            fontsize=11, color='#A23B72', fontweight='bold')

plt.xlabel('Number of Components (k)', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative Variance Explained (%)', fontsize=14, fontweight='bold')
plt.title('Comparison: Temporal Correlation vs V-Domain PCA\nCumulative Variance Explained', 
          fontsize=16, fontweight='bold', pad=20)

plt.xscale('log')
plt.xlim(0.8, 1200)
plt.ylim(0, 90)

plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12, loc='center right', frameon=True, fancybox=True, shadow=True)

textstr = 'Temporal PCA captures 51% variance\nwith first component alone!'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

plt.tight_layout()

plt.savefig(os.path.join(output_dir, "combined_variance_explained.png"), 
            dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("Saved combined variance explained plot to output directory.")


# === Top 16 Spatial Modes 
num_spatial_modes = 16
modes_to_plot = pca.components_[:num_spatial_modes]
modes_imgs = modes_to_plot.reshape(num_spatial_modes, H, W)

fig, axes = plt.subplots(2, 8, figsize=(16, 8))
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i < num_spatial_modes:
        img = normalize_to_255(modes_imgs[i])
        ax.imshow(img ,cmap='gray')
        ax.set_title(f"Mode {i+1}", fontsize=12)
        ax.axis("off")
    else:
        ax.axis("off")

plt.suptitle("First 16 Spatial Modes", fontsize=18, weight="bold")

plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, 
                    wspace=-0.05, 
                    hspace=0.15)   
plt.savefig(os.path.join(output_dir, "top12_spatial_modes_ultracompact.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved top 12 spatial modes in ultra-compact layout.")


# === Seismic-style PCA temporal components  ===
num_modes_req = 25
base_sep = 50
band_frac = 0.6
time_axis = np.arange(T) / fps

num_modes = min(num_modes_req, X_full_pca.shape[1])
temporal_modes = X_full_pca[:, :num_modes]
eigenvalues = pca_full.explained_variance_[:num_modes]

scaled_modes = temporal_modes * np.sqrt(eigenvalues)
p2p = scaled_modes.max(axis=0) - scaled_modes.min(axis=0)
scaled_modes = (scaled_modes / (p2p + 1e-12)) * (band_frac * base_sep)

offsets = np.arange(num_modes, 0, -1) * base_sep
stacked_modes = scaled_modes + offsets[np.newaxis, :]

fig, ax = plt.subplots(figsize=(18, 10))  # larger
colors = plt.cm.viridis(np.linspace(0.3, 1, num_modes))

for i in range(num_modes):
    ax.plot(time_axis, stacked_modes[:, i], lw=0.8, color=colors[i], alpha=0.9)
    ax.axhline(offsets[i], ls=":", lw=0.6, color="black")  # mean line
    ax.text(time_axis[0] - 0.5, offsets[i], f"PC{i+1}", va="center", ha="right", fontsize=9)

ax.set_xlim(time_axis[0], time_axis[-1])
ax.set_xlabel("Time (seconds)")
ax.set_title(f"Seismic-style PCA Temporal Components (Top {num_modes})", fontsize=16, weight="bold")

ax.yaxis.set_visible(False)
for spine in ("left", "right", "top"):
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "temporal_components_seismic_fullpage.png"), dpi=300)
plt.show()
plt.close()
print("Saved improved temporal PCA seismic plot.")


#Side-by-side comparison 
frames_combo = []
spacer_width = 10
label_font = cv2.FONT_HERSHEY_SIMPLEX
label_scale = 0.6
label_color = (255, 255, 255)  # white text
label_thick = 2

for t in range(T):
    orig_frame = video[t]

  
    recon_frames = all_recons[k][t]
    
    rotated_H = orig_frame.shape[0]
    spacer = np.ones((rotated_H, spacer_width, 3), dtype=np.uint8) * 255
    orig_labeled = orig_frame.copy()
    cv2.putText(orig_labeled, "Original", (10, 20), label_font, label_scale, label_color, label_thick)
    
    recon_labeled = []
    for idx, f in enumerate(recon_frames):
        f_copy = f.copy()
        cv2.putText(f_copy, f"k={k_values[idx]}", (10, 20), label_font, label_scale, label_color, label_thick)
        recon_labeled.append(f_copy)
    
    row = orig_labeled
    for f in recon_labeled:
        row = np.hstack([row, spacer, f])
    
    frames_combo.append(row)

rotated_h, rotated_w, _ = frames_combo[0].shape  # swap height/width automatically
out = cv2.VideoWriter(
    os.path.join(output_dir, "comparison_video_all_k_rotated.mp4"),
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (rotated_w, rotated_h)
)
for frame in frames_combo:
    out.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
out.release()
print("Saved labeled comparison video (rotated to portrait).")


#Comparison Grid

compare_k_values = [5, 20, 50, 100, 200]
compare_times = [T // 4, T // 2, 3 * T // 4]  
num_rows = len(compare_times)
num_cols = len(compare_k_values) + 1  
fps = get_video_fps(video_path)

fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))  

for row, t in enumerate(compare_times):
    orig_frame = video[t].astype(np.uint8)   
    axs[row, 0].imshow(orig_frame)
    axs[row, 0].axis("off")
    
    seconds = t / fps
    timestamp = f"{int(seconds//60):02d}:{int(seconds%60):02d}"  
    axs[row, 0].set_title(f"Original\n{timestamp}", fontsize=10)

    for col, k in enumerate(compare_k_values):
        pca_k = PCA(n_components=k)
        X_k = pca_k.fit_transform(X)
        Xk_reconstructed = pca_k.inverse_transform(X_k)
        recon_frame = Xk_reconstructed[t].reshape(H, W) * std_image + mean_image
        rgb_flat = (recon_frame.flatten()[:, None] * dominant_vec[None, :]) + mean_rgb
        rgb_frame = np.clip(rgb_flat.reshape(H, W, 3), 0, 255).astype(np.uint8)
        axs[row, col + 1].imshow(rgb_frame)
        axs[row, col + 1].axis("off")
        if row == 0:
            axs[row, col + 1].set_title(f"k={k}", fontsize=10)

plt.suptitle("Original vs. Reconstructed Frames", fontsize=16)

plt.tight_layout()
plt.subplots_adjust(top=0.88, wspace=0.01)  

plt.savefig(os.path.join(output_dir, "frame_comparison_grid_timestamp.png"), dpi=300)
plt.close()
print("Saved frame comparison grid with timestamps for original frames.")



# === RGB CENTER PIXEL OVER TIME ===

center_h, center_w = H // 2, W // 2
rgb_center = video[:, center_h, center_w, :]  # shape (T, 3)

plt.figure(figsize=(10, 5))
colors = ['red', 'green', 'blue']
for i, color in enumerate(colors):
    plt.plot(rgb_center[:, i], label=f"{color.upper()} Channel", color=color)
plt.title("RGB Intensity at Center Pixel Over Time")
plt.xlabel("Frame")
plt.ylabel("Intensity")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "center_pixel_rgb.png"))
plt.close()
print("Saved RGB center pixel time series.")


#10 Random Pixel Time Series

import numpy as np
import matplotlib.pyplot as plt


pixel_coords = [(20, 30), (40, 50), (60, 80),
                (80, 100), (100, 120), (40, 200),
                (60, 220), (100, 180), (120, 140), (70, 160)]   #

ts_list = [video[:, r, c, 2] for (r, c) in pixel_coords]
ts_array = np.array(ts_list)  

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

sep = 255
for i, ts in enumerate(ts_array):
    ts_centered = ts - np.mean(ts)
    axs[0].plot(ts_centered + i*sep, color='black', lw=0.8)
    axs[0].axhline(i*sep, color='gray', linestyle='--', linewidth=0.5)

axs[0].set_title("Seismic-style Time Series")
axs[0].set_xlabel("Frame")
axs[0].set_ylabel("Pixel signals ")
axs[0].set_yticks([i*sep for i in range(len(pixel_coords))])
axs[0].set_yticklabels([f"Pixel {i+1}" for i in range(len(pixel_coords))])

frame0 = video[0, :, :, 2]
axs[1].imshow(frame0, cmap='gray')
axs[1].set_title("Pixel Locations")
axs[1].axis('off')

rot_coords = [(c, frame0.shape[0]-r-1) for (r, c) in pixel_coords]
for i, (r, c) in enumerate(rot_coords):
    axs[1].plot(c, r, 'o', color='white', markersize=10, 
                markeredgecolor='black', markeredgewidth=2)
    axs[1].text(c+2, r, f"{i+1}", color="black", fontsize=9, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                         edgecolor='black', alpha=0.8))

plt.tight_layout()
plt.show()








