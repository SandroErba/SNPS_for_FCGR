# produce_fcgr.R (Processes MAX 10 samples per genus with real-time feedback)
library(kaos)
library(seqinr)
library(foreach)
library(doParallel)

# Setup parallel cluster
num_cores <- min(10, detectCores())
cl <- makeCluster(num_cores)
registerDoParallel(cl)

genera <- c("Aeromonas", "Citrobacter", "Helicobacter", "Streptococcus")
MAX_SAMPLES <- 10

cat(paste0("[+] Starting FCGR matrix extraction (28x28) for MAX ", MAX_SAMPLES, " samples per genus using ", num_cores, " CPU cores...\n"))

for (g in genera) {
  input_dir <- paste0("ncbi_dataset_", g, "/ncbi_dataset/data")
  output_dir <- paste0("dataset_28x28/", g)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  folders <- list.files(input_dir, pattern = "^GC[FA]", full.names = TRUE)
  
  # Strictly limit to MAX_SAMPLES per genus
  folders <- head(folders, MAX_SAMPLES)
  total_samples <- length(folders)
  
  cat(paste0("\n----------------------------------------------------------\n"))
  cat(paste0(" Processing genus: ", g, " (Targeting ", total_samples, " samples)\n"))
  cat(paste0("----------------------------------------------------------\n"))
  
  if (total_samples == 0) {
    cat(paste0("[!] Warning: No genome folders found in ", input_dir, "\n"))
    next
  }
  
  # Print upfront task queue so you know the script is active immediately
  for (k in 1:total_samples) {
    folder_name <- tail(unlist(strsplit(folders[k], split = '/')), n = 1)
    cat(paste0(" -> Queued sample ", k, "/", total_samples, ": ", folder_name, "\n"))
  }
  cat("[+] Executing parallel extraction on cluster...\n")
  
  results <- foreach(i = 1:total_samples, .packages = c('kaos', 'seqinr'), .combine = 'c') %dopar% {
    folder_name <- tail(unlist(strsplit(folders[i], split = '/')), n = 1)
    out_file <- paste0(output_dir, "/fcgr_", folder_name, ".txt")
    
    if (!file.exists(out_file)) {
      fasta_files <- list.files(folders[i], pattern = "\\.(fna|fa|fasta)$", full.names = TRUE)
      
      if (length(fasta_files) > 0) {
        seq <- read.fasta(fasta_files[1], as.string = TRUE)
        seq_chars <- unlist(strsplit(seq[[1]], split = ''))
        
        # Direct 28x28 grid discretization
        cgr_list <- cgr(seq_chars, res = 28)
        
        write.table(cgr_list$matrix, file = out_file, row.names = FALSE, quote = FALSE, col.names = FALSE)
        return(paste0("[OK] Completed sample ", i, "/", total_samples, ": ", folder_name))
      } else {
        return(paste0("[SKIP] No FASTA found for ", folder_name))
      }
    } else {
      return(paste0("[EXISTS] Sample ", i, "/", total_samples, " already processed."))
    }
  }
  
  # Output worker execution summary
  for (res in results) {
    cat(paste0(res, "\n"))
  }
}

stopCluster(cl)
cat("\n==========================================================\n")
cat("[+] FCGR 28x28 extraction completed successfully for all genera.\n")
cat("==========================================================\n")