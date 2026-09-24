#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
base_dir <- if (length(args) >= 1) normalizePath(args[1]) else normalizePath(".")
out_dir <- file.path(base_dir, "analysis_plots")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

result_files <- Sys.glob(file.path(base_dir, "seed_*", "S_*", "run_*", "task_result.csv"))
if (length(result_files) == 0) stop("No task_result.csv files found under ", base_dir)

records <- lapply(result_files, function(result_file) {
  run_dir <- dirname(result_file)
  parts <- strsplit(normalizePath(result_file), "/", fixed = TRUE)[[1]]
  seed <- parts[grepl("^seed_", parts)]
  spec <- parts[grepl("^S_", parts)]
  coverage_file <- file.path(run_dir, "coverage_timeseries.csv")
  path_file <- file.path(run_dir, "coverage_paths.csv")
  bumps_file <- file.path(run_dir, "bumps_global.csv")

  coverage_data <- read.csv(coverage_file)
  final_coverage <- tail(coverage_data$coverage_pct, 1)

  paths <- read.csv(path_file)
  total_path <- sum(vapply(split(paths, paths$robot), function(robot_path) {
    if (nrow(robot_path) < 2) return(0)
    sum(sqrt(diff(robot_path$x)^2 + diff(robot_path$y)^2))
  }, numeric(1)))

  bumps <- read.csv(bumps_file)
  data.frame(
    seed = seed,
    spec = spec,
    coverage = final_coverage,
    path_length = total_path,
    bumps = nrow(bumps),
    stringsAsFactors = FALSE
  )
})
results <- do.call(rbind, records)

spec_stats <- aggregate(coverage ~ spec, results, function(x) {
  c(mean = mean(x), sd = sd(x), se = sd(x) / sqrt(length(x)), n = length(x))
})
spec_stats <- data.frame(
  spec = spec_stats$spec,
  mean = spec_stats$coverage[, "mean"],
  sd = spec_stats$coverage[, "sd"],
  se = spec_stats$coverage[, "se"],
  n = spec_stats$coverage[, "n"]
)
spec_stats$short_spec <- sub("^S_20260912_", "", spec_stats$spec)
spec_stats <- spec_stats[order(spec_stats$mean), ]
write.csv(spec_stats[order(-spec_stats$mean), ], file.path(out_dir, "specification_summary.csv"), row.names = FALSE)

png(file.path(out_dir, "01_specification_ranking.png"), width = 1800, height = 1400, res = 180)
par(mar = c(5, 8, 3, 2))
ci <- 2.262 * spec_stats$se
xlim <- range(c(spec_stats$mean - ci, spec_stats$mean + ci, 45, 95))
bp <- barplot(spec_stats$mean, names.arg = spec_stats$short_spec, horiz = TRUE,
              las = 1, col = ifelse(spec_stats$spec == "S_20260912_105245", "#E69F00", "#4C78A8"),
              border = NA, xlim = xlim, xlab = "Final coverage (%)",
              main = "Fixed-specification coverage ranking (10 seeds each)")
arrows(spec_stats$mean - ci, bp, spec_stats$mean + ci, bp,
       angle = 90, code = 3, length = 0.03, col = "#333333", lwd = 1.2)
abline(v = mean(results$coverage), lty = 2, col = "#D62728", lwd = 2)
legend("bottomright", c("Best mean", "Overall mean", "Approx. 95% CI"),
       fill = c("#E69F00", NA, NA), border = c(NA, NA, NA),
       lty = c(NA, 2, 1), col = c(NA, "#D62728", "#333333"), bty = "n")
dev.off()

spec_order <- spec_stats$spec[order(-spec_stats$mean)]
seed_order <- sort(unique(results$seed))
heat <- matrix(NA_real_, nrow = length(spec_order), ncol = length(seed_order),
               dimnames = list(sub("^S_20260912_", "", spec_order), sub("seed_", "", seed_order)))
for (i in seq_len(nrow(results))) {
  heat[sub("^S_20260912_", "", results$spec[i]), sub("seed_", "", results$seed[i])] <- results$coverage[i]
}
png(file.path(out_dir, "02_coverage_heatmap.png"), width = 1800, height = 1500, res = 180)
par(mar = c(5, 7, 4, 6))
palette <- colorRampPalette(c("#B2182B", "#FDDDBC", "#D1E5F0", "#2166AC"))(100)
image(seq_len(ncol(heat)), seq_len(nrow(heat)), t(heat[nrow(heat):1, ]),
      col = palette, zlim = c(45, 100), axes = FALSE,
      xlab = "Seed", ylab = "Specification (ranked by mean coverage)",
      main = "Final coverage by specification and seed")
axis(1, at = seq_len(ncol(heat)), labels = colnames(heat))
axis(2, at = seq_len(nrow(heat)), labels = rev(rownames(heat)), las = 1, cex.axis = 0.7)
box()
par(new = TRUE, fig = c(0.91, 0.94, 0.18, 0.82), mar = c(0, 0, 0, 0))
image(1, seq(45, 100, length.out = 100), matrix(seq(45, 100, length.out = 100), nrow = 1),
      col = palette, axes = FALSE, xlab = "", ylab = "")
axis(4, at = seq(45, 100, by = 10), las = 1)
mtext("Coverage (%)", side = 4, line = 2.5)
dev.off()

png(file.path(out_dir, "03_coverage_distribution.png"), width = 1600, height = 1000, res = 180)
par(mar = c(5, 5, 3, 2))
hist(results$coverage, breaks = seq(40, 100, by = 5), col = "#4C78A8", border = "white",
     xlab = "Final coverage (%)", ylab = "Number of runs",
     main = "Distribution of final coverage across 300 runs")
abline(v = mean(results$coverage), col = "#D62728", lwd = 2, lty = 2)
abline(v = median(results$coverage), col = "#E69F00", lwd = 2, lty = 3)
legend("topleft", c(sprintf("Mean: %.1f%%", mean(results$coverage)),
                    sprintf("Median: %.1f%%", median(results$coverage))),
       col = c("#D62728", "#E69F00"), lty = c(2, 3), lwd = 2, bty = "n")
dev.off()

png(file.path(out_dir, "04_coverage_relationships.png"), width = 1800, height = 850, res = 180)
par(mfrow = c(1, 2), mar = c(5, 5, 3, 2))
plot(results$path_length, results$coverage, pch = 19, col = rgb(0.15, 0.45, 0.70, 0.5),
     xlab = "Combined robot path length (m)", ylab = "Final coverage (%)",
     main = sprintf("Coverage vs path length (r = %.2f)", cor(results$coverage, results$path_length)))
abline(lm(coverage ~ path_length, results), col = "#D62728", lwd = 2)
jittered_bumps <- jitter(results$bumps, amount = 0.08)
plot(jittered_bumps, results$coverage, pch = 19, col = rgb(0.90, 0.45, 0.10, 0.45),
     xaxt = "n", xlab = "Recorded bumps", ylab = "Final coverage (%)",
     main = sprintf("Coverage vs bumps (r = %.2f)", cor(results$coverage, results$bumps)))
axis(1, at = 0:3)
abline(lm(coverage ~ bumps, results), col = "#D62728", lwd = 2)
dev.off()

message("Created plots in: ", out_dir)
