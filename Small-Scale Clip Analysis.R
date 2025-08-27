library(av)
library(magick)
library(png)
library(ggplot2)

video_path <- "C:/Users/l/Downloads/waveclip.mp4"
output_dir <- "C:/Users/l/Downloads/wave_pca_complete_output"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
fps <- 30
k_values <- c(1, 5, 10, 20, 50, 100)

# ==== HELPER FUNCTIONS ====
read_frames <- function(video_path) {
  temp_dir <- tempfile("frames")
  dir.create(temp_dir)
  av::av_video_images(video_path, destdir = temp_dir, format = "png")
  frame_files <- list.files(temp_dir, full.names = TRUE, pattern = "\\.png$")
  lapply(frame_files, image_read)
}

flatten_frame <- function(img) {
  data <- image_data(img, channels = "rgb")
  r <- as.numeric(data[1, , ])
  g <- as.numeric(data[2, , ])
  b <- as.numeric(data[3, , ])
  
  gamma <- 1.8
  r <- ((r / 255) ^ gamma) * 255
  g <- ((g / 255) ^ gamma) * 255
  b <- ((b / 255) ^ gamma) * 255
  
  c(r, g, b)
}

reconstruct_frame <- function(vec, height, width) {
  n_pix <- height * width
  r_vec <- vec[1:n_pix]
  g_vec <- vec[(n_pix + 1):(2 * n_pix)]
  b_vec <- vec[(2 * n_pix + 1):(3 * n_pix)]
  
  gamma <- 1.8
  r_vec <- ((pmax(0, pmin(255, r_vec)) / 255) ^ (1/gamma)) * 255
  g_vec <- ((pmax(0, pmin(255, g_vec)) / 255) ^ (1/gamma)) * 255
  b_vec <- ((pmax(0, pmin(255, b_vec)) / 255) ^ (1/gamma)) * 255
  
  r_vec <- round(r_vec)
  g_vec <- round(g_vec)
  b_vec <- round(b_vec)
  
  r_mat <- matrix(r_vec, nrow = width, ncol = height)
  g_mat <- matrix(g_vec, nrow = width, ncol = height)
  b_mat <- matrix(b_vec, nrow = width, ncol = height)
  
  temp_file <- tempfile(fileext = ".png")
  
  img_array <- array(0, dim = c(height, width, 3))
  img_array[,,1] <- t(r_mat) / 255  # PNG expects values 0-1, not 0-255
  img_array[,,2] <- t(g_mat) / 255
  img_array[,,3] <- t(b_mat) / 255
  
  png::writePNG(img_array, temp_file)
  image_read(temp_file)
}

# ==== MAIN PROCESSING ====
cat("Reading frames from video...\n")
imgs <- read_frames(video_path)
n_frames <- length(imgs)

info <- image_info(imgs[[1]])
height <- info$height
width <- info$width

cat("Loaded", n_frames, "frames of size", width, "x", height, "\n")


# Flatten all frames
cat("Flattening all frames for PCA...\n")
data_mat <- t(do.call(cbind, lapply(imgs, flatten_frame)))  # frames x (pixels*3)
cat("Data matrix dimensions:", dim(data_mat), "\n")

# Perform PCA
cat("Performing PCA...\n")
pca_res <- prcomp(data_mat, center = TRUE, scale. = FALSE)

# Calculate variance explained
var_explained <- (pca_res$sdev)^2
total_var <- sum(var_explained)
cumvar <- cumsum(var_explained) / total_var * 100
individual_var <- var_explained / total_var * 100

# Print variance explained for k=1 to 10
cat("\n=== VARIANCE EXPLAINED BY COMPONENTS ===\n")
for (k in 1:min(10, length(individual_var))) {
  cat(sprintf("k=%2d: Individual = %6.2f%%, Cumulative = %6.2f%%\n", 
              k, individual_var[k], cumvar[k]))
}
cat("\n")

# Create variance explained plot
cat("Creating variance explained plot...\n")
plot_data <- data.frame(
  Component = 1:min(200, length(cumvar)),
  Cumulative_Variance = cumvar[1:min(200, length(cumvar))]
)

p <- ggplot(plot_data, aes(x = Component, y = Cumulative_Variance)) +
  geom_line(size = 1, color = "blue") +
  geom_point(size = 2, color = "darkblue") +
  geom_hline(yintercept = c(50, 80, 90, 95), 
             linetype = "dashed", 
             color = c("green", "orange", "red", "purple"),
             alpha = 0.7) +
  annotate("text", x = max(plot_data$Component) * 0.8, y = c(52, 82, 92, 97), 
           label = c("50%", "80%", "90%", "95%"), 
           color = c("green", "orange", "red", "purple")) +
  labs(title = "PCA Cumulative Variance Explained",
       subtitle = paste("Video:", basename(video_path), "| Frames:", n_frames),
       x = "Number of Principal Components",
       y = "Cumulative Variance Explained (%)") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 10))

ggsave(file.path(output_dir, "variance_explained_plot.png"), 
       plot = p, width = 10, height = 6, dpi = 300)


# ==== VIDEO RECONSTRUCTION FUNCTION ====
reconstruct_video <- function(k) {
  cat(sprintf("=== RECONSTRUCTING WITH k=%d COMPONENTS ===\n", k))
  k_actual <- min(k, ncol(pca_res$x))
  if (k_actual != k) {
    cat("Warning: Reduced k from", k, "to", k_actual, "(max available)\n")
  }
  
  scores <- pca_res$x[, 1:k_actual, drop = FALSE]
  loadings <- pca_res$rotation[, 1:k_actual, drop = FALSE]
  
  recon <- scores %*% t(loadings)
  recon <- sweep(recon, 2, pca_res$center, "+")
  
  output_dir_k <- file.path(output_dir, sprintf("frames_k%03d", k))
  dir.create(output_dir_k, showWarnings = FALSE)
  cat("Reconstructing", n_frames, "frames...\n")
  start_time <- Sys.time()
  
  for (i in 1:n_frames) {
    frame_vec <- recon[i, ]
    frame_img <- reconstruct_frame(frame_vec, height, width)
    
    output_path <- file.path(output_dir_k, sprintf("frame_%04d.png", i))
    image_write(frame_img, path = output_path)
    
    if (i %% 20 == 0 || i == n_frames) {
      elapsed <- as.numeric(Sys.time() - start_time, units = "secs")
      cat(sprintf("  Frame %d/%d (%.1f%%) - %.1fs elapsed\n", 
                  i, n_frames, 100*i/n_frames, elapsed))
    }
  }
  
  # Create video
  cat("Encoding video...\n")
  png_files <- list.files(output_dir_k, pattern = "\\.png$", full.names = TRUE)
  png_files <- sort(png_files)  
  
  output_video <- file.path(output_dir, sprintf("reconstructed_k%03d.mp4", k))
  av_encode_video(png_files, output = output_video, framerate = fps)
  return(list(k = k, variance_explained = sum(individual_var_pct[1:k])))

  

# ==== RUN ALL RECONSTRUCTIONS ====
cat("Starting video reconstructions...\n")
total_start_time <- Sys.time()
reconstruction_results <- list()

for (k in k_values) {
  if (k <= min(nrow(data_mat), ncol(data_mat))) {
    results <- reconstruct_video(k)
    reconstruction_results[[length(reconstruction_results) + 1]] <- results
  } else {
    cat(sprintf("Skipping k=%d (exceeds %d available components)\n\n", 
                k, min(nrow(data_mat), ncol(data_mat))))
  }
}

total_time <- as.numeric(Sys.time() - total_start_time, units = "mins")

cat("========================================\n")
cat("              SUMMARY\n") 
cat("========================================\n")
cat(sprintf("Video: %s\n", basename(video_path)))
cat(sprintf("Frames: %d | Dimensions: %dx%d\n", n_frames, width, height))
cat(sprintf("Total processing time: %.1f minutes\n\n", total_time))

cat("RECONSTRUCTIONS COMPLETED:\n")
for (result in reconstruction_results) {
  cat(sprintf("  k=%3d: %6.2f%% variance explained\n", 
              result$k, result$variance_explained))
}

cat("\nOUTPUT FILES:\n")
cat(sprintf("  - Variance plots: %s\n", file.path(output_dir, "variance_explained_*.png")))
cat(sprintf("  - Videos: %s\n", file.path(output_dir, "reconstructed_k*.mp4")))
cat(sprintf("  - Frame folders: %s\n", file.path(output_dir, "frames_k*/")))
cat("========================================\n")



# Calculate individual variance explained by each PC
individual_variance <- (pca_res$sdev)^2
total_variance <- sum(individual_variance)
individual_var_pct <- (individual_variance / total_variance) * 100

# Create detailed table for PC 1-10
cat("\n📊 INDIVIDUAL VARIANCE EXPLAINED BY PRINCIPAL COMPONENTS:\n")
cat("=" %in% paste(rep("=", 65), collapse = ""), "\n")
cat(sprintf("%-4s | %-15s | %-15s | %-15s\n", 
            "PC", "Eigenvalue", "Variance %", "Cumulative %"))
cat("-" %in% paste(rep("-", 65), collapse = ""), "\n")

cumulative_var <- cumsum(individual_var_pct)

for (i in 1:min(10, length(individual_var_pct))) {
  cat(sprintf("%-4d | %-15.4f | %-15.2f | %-15.2f\n", 
              i, 
              individual_variance[i], 
              individual_var_pct[i], 
              cumulative_var[i]))
}

cat("=" %in% paste(rep("=", 65), collapse = ""), "\n")

pc_variance_table <- data.frame(
  PC = 1:min(10, length(individual_var_pct)),
  Eigenvalue = individual_variance[1:min(10, length(individual_var_pct))],
  Variance_Percent = individual_var_pct[1:min(10, length(individual_var_pct))],
  Cumulative_Percent = cumulative_var[1:min(10, length(individual_var_pct))]
)

csv_path <- file.path(output_dir, "pc_variance_explained.csv")
write.csv(pc_variance_table, csv_path, row.names = FALSE)

cat(sprintf("\n✓ Variance table saved: %s\n", csv_path))
cat("✓ Individual PC variance analysis complete!\n")



