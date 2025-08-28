library(av)
library(magick)

# ==== CONFIG ====
video_path <- "C:/Users/l/Downloads/waveclip.mp4"
output_dir <- "C:/Users/l/Downloads/wave_pca_reconstruction"
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
  img_array[,,1] <- t(r_mat) / 255
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
cat("Loaded", n_frames, "frames of size", width, "x", height, "\n\n")

# Flatten frames
data_mat <- t(do.call(cbind, lapply(imgs, flatten_frame)))

# Perform PCA
cat("Performing PCA...\n")
pca_res <- prcomp(data_mat, center = TRUE, scale. = FALSE)
var_explained <- (pca_res$sdev)^2
cumvar <- cumsum(var_explained) / sum(var_explained) * 100

# ==== VIDEO RECONSTRUCTION FUNCTION ====
reconstruct_video <- function(k) {
  k_actual <- min(k, ncol(pca_res$x))
  scores <- pca_res$x[, 1:k_actual, drop = FALSE]
  loadings <- pca_res$rotation[, 1:k_actual, drop = FALSE]
  
  recon <- scores %*% t(loadings)
  recon <- sweep(recon, 2, pca_res$center, "+")
  
  output_dir_k <- file.path(output_dir, sprintf("frames_k%03d", k_actual))
  dir.create(output_dir_k, showWarnings = FALSE)
  
  for (i in 1:n_frames) {
    frame_vec <- recon[i, ]
    frame_img <- reconstruct_frame(frame_vec, height, width)
    output_path <- file.path(output_dir_k, sprintf("frame_%04d.png", i))
    image_write(frame_img, path = output_path)
  }
  
  png_files <- list.files(output_dir_k, pattern = "\\.png$", full.names = TRUE)
  png_files <- sort(png_files)
  
  output_video <- file.path(output_dir, sprintf("reconstructed_k%03d.mp4", k_actual))
  av_encode_video(png_files, output = output_video, framerate = fps)
  
  cat(sprintf("Video saved: %s | Variance explained: %.2f%%\n", 
              basename(output_video), ifelse(k_actual <= length(cumvar), cumvar[k_actual], 100)))
}

# ==== RUN RECONSTRUCTIONS ====
cat("Starting video reconstructions...\n")
for (k in k_values) {
  if (k <= min(nrow(data_mat), ncol(data_mat))) {
    reconstruct_video(k)
  } else {
    cat(sprintf("Skipping k=%d (exceeds max available components)\n", k))
  }
}

cat("\n Reconstructed videos saved in:\n", output_dir, "\n")
