# produce_fcgr.R (Fast Single-Pass Multi-Resolution Extraction)
library(kaos)
library(seqinr)
library(foreach)
library(doParallel)

# ==========================================================
# CONFIGURATION PARAMETERS
# ==========================================================
GRID_SIZES <- c(4, 8, 16, 32)  # Resolutions to generate in 1 pass
MAX_SAMPLES <- 10                   # Max samples per genus
# ==========================================================

# Setup parallel cluster
num_cores <- min(10, detectCores())
cl <- makeCluster(num_cores)
registerDoParallel(cl)

genera <- c("Aeromonas", "Citrobacter", "Helicobacter", "Streptococcus")

# Pre-create all dataset folders
for (res in GRID_SIZES) {
  for (g in genera) {
    dir.create(paste0("dataset_", res, "x", res, "/", g), recursive = TRUE, showWarnings = FALSE)
  }
}

cat(paste0("[+] Starting Single-Pass FCGR extraction\n"))
cat(paste0("[+] Resolving sizes: ", paste(GRID_SIZES, collapse = ", "), " simultaneously using ", num_cores, " CPU cores...\n"))

for (g in genera) {
  input_dir <- paste0("ncbi_dataset_", g, "/ncbi_dataset/data")
  folders <- head(list.files(input_dir, pattern = "^GC[FA]", full.names = TRUE), MAX_SAMPLES)
  total_samples <- length(folders)
  
  cat(paste0("\n----------------------------------------------------------\n"))
  cat(paste0(" Processing genus: ", g, " (", total_samples, " samples) across all resolutions\n"))
  cat(paste0("----------------------------------------------------------\n"))
  
  if (total_samples == 0) next

  results <- foreach(i = 1:total_samples, .packages = c('kaos', 'seqinr'), .combine = 'c') %dopar% {
    folder_name <- tail(unlist(strsplit(folders[i], split = '/')), n = 1)
    fasta_files <- list.files(folders[i], pattern = "\\.(fna|fa|fasta)$", full.names = TRUE)
    
    if (length(fasta_files) > 0) {
      # 1. READ FASTA ONCE INTO MEMORY
      seq <- read.fasta(fasta_files[1], as.string = TRUE)
      seq_chars <- unlist(strsplit(seq[[1]], split = ''))
      
      # 2. GENERATE ALL MATRICES FROM THE SAME IN-MEMORY SEQUENCE
      for (GRID_SIZE in GRID_SIZES) {
        output_dir <- paste0("dataset_", GRID_SIZE, "x", GRID_SIZE, "/", g)
        out_file <- paste0(output_dir, "/fcgr_", folder_name, ".txt")
        
        if (!file.exists(out_file)) {
          cgr_list <- cgr(seq_chars, res = GRID_SIZE)
          write.table(cgr_list$matrix, file = out_file, row.names = FALSE, quote = FALSE, col.names = FALSE)
        }
      }
      return(paste0("[OK] Processed sample ", i, "/", total_samples, ": ", folder_name, " (All grid sizes saved)"))
    } else {
      return(paste0("[SKIP] No FASTA found for ", folder_name))
    }
  }
  
  for (res in results) {
    cat(paste0(res, "\n"))
  }
}

stopCluster(cl)
cat("\n==========================================================\n")
cat("[+] ALL RESOLUTIONS GENERATED\n")
cat("==========================================================\n")
