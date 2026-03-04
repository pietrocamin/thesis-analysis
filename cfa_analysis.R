# =============================================================================
# CONFIRMATORY FACTOR ANALYSIS (CFA)
# Scale: Well-Being in Human-Robot Interaction
# Estimator: WLSMV (robust, appropriate for ordinal Likert data)
# Library: lavaan
# =============================================================================
# Models tested:
#   Model 1 - Theoretical: four factors as originally hypothesised
#             (Competence, Autonomy, HH-Relatedness, HR-Relatedness)
#             with a higher-order Well-Being factor
#   Model 2 - Empirical: four factors derived from EFA solution
#             with a higher-order Well-Being factor
#   Model 3 - Bifactor: general WB factor + four group factors simultaneously
#             (excluded if Heywood cases detected)
# =============================================================================

# --- 0. Setup -----------------------------------------------------------------

packages <- c("lavaan", "semTools", "dplyr", "readr", "tibble", "ggplot2")
installed <- packages %in% rownames(installed.packages())
if (any(!installed)) {
  install.packages(packages[!installed], repos = "https://cloud.r-project.org")
}

library(lavaan)
library(semTools)
library(dplyr)
library(readr)
library(tibble)
library(ggplot2)

# --- 1. Load Data -------------------------------------------------------------

file_path <- "DATASETS/reversed_DATASET_problematic_items_clean.csv"

if (!file.exists(file_path)) {
  stop(paste("Error: File not found at", file_path))
}

df <- read_csv(file_path, show_col_types = FALSE)

# Items used (20 items after removal of A4 and HRR3)
items <- c("C1", "C2", "C3", "C4", "C5", "C6", "C7",
           "A1", "A2", "A3", "A5", "A6", "A7",
           "HHR1", "HHR2", "HHR3", "HHR4",
           "HRR1", "HRR2", "HRR4")

# Treat items as ordered factors (required for WLSMV / polychoric correlations)
df_ord <- df %>%
  select(all_of(items)) %>%
  mutate(across(everything(), ~ ordered(.)))

cat("Data loaded successfully.\n")
cat(sprintf("Sample size: %d participants, %d items\n\n", nrow(df_ord), length(items)))


# =============================================================================
# 2. MODEL SPECIFICATIONS
# =============================================================================

# -----------------------------------------------------------------------------
# MODEL 1: Theoretically specified higher-order model
# Four first-order factors defined by original subscale assignment.
# One second-order factor: Well-Being.
# All cross-loadings fixed to zero (strict theoretical constraints).
# -----------------------------------------------------------------------------

model1 <- '
  Competence =~ C1 + C2 + C3 + C4 + C5 + C6 + C7
  Autonomy   =~ A1 + A2 + A3 + A5 + A6 + A7
  HHR        =~ HHR1 + HHR2 + HHR3 + HHR4
  HRR        =~ HRR1 + HRR2 + HRR4
  WellBeing  =~ Competence + Autonomy + HHR + HRR
'

# -----------------------------------------------------------------------------
# MODEL 2: Empirically informed higher-order model (EFA 4-factor solution).
# F1 (Self-Efficacy)       = C1, C2, C3, C4, A1, A5
# F2 (Social Relatedness)  = A6, A7, HHR1, HHR2, HRR1, HRR2, HRR4
# F3 (Evaluative/Method)   = C5, C6, C7, HHR3, HHR4
# F4 (External Regulation) = A2, A3
# Note: A2 loading > 1.0 in output is a Heywood-adjacent instability
# attributable to F4 having only 2 indicators. Report as a limitation.
# -----------------------------------------------------------------------------

model2 <- '
  F1_SelfEfficacy  =~ C1 + C2 + C3 + C4 + A1 + A5
  F2_SocialRelated =~ A6 + A7 + HHR1 + HHR2 + HRR1 + HRR2 + HRR4
  F3_Evaluative    =~ C5 + C6 + C7 + HHR3 + HHR4
  F4_ExternalReg   =~ A2 + A3
  WellBeing =~ F1_SelfEfficacy + F2_SocialRelated + F3_Evaluative + F4_ExternalReg
'

# -----------------------------------------------------------------------------
# MODEL 3: Bifactor model.
# Each item loads simultaneously on a general WB g-factor and its group factor.
# Group factors and g-factor are orthogonal (standard bifactor specification).
# Non-convergence (Heywood cases) will exclude this model automatically.
# -----------------------------------------------------------------------------

model3 <- '
  g =~ C1 + C2 + C3 + C4 + C5 + C6 + C7 +
       A1 + A2 + A3 + A5 + A6 + A7 +
       HHR1 + HHR2 + HHR3 + HHR4 +
       HRR1 + HRR2 + HRR4
  Competence =~ C1 + C2 + C3 + C4 + C5 + C6 + C7
  Autonomy   =~ A1 + A2 + A3 + A5 + A6 + A7
  HHR        =~ HHR1 + HHR2 + HHR3 + HHR4
  HRR        =~ HRR1 + HRR2 + HRR4
  g          ~~ 0*Competence
  g          ~~ 0*Autonomy
  g          ~~ 0*HHR
  g          ~~ 0*HRR
  Competence ~~ 0*Autonomy
  Competence ~~ 0*HHR
  Competence ~~ 0*HRR
  Autonomy   ~~ 0*HHR
  Autonomy   ~~ 0*HRR
  HHR        ~~ 0*HRR
'


# -----------------------------------------------------------------------------
# MODEL 4: Two-factor correlated model (exploratory).
# Motivated by PA and MAP converging on 2 factors, and by the very high
# inter-factor correlations in Model 1 (Competence-Autonomy = 0.806),
# which suggest these constructs may not be empirically distinguishable.
#
# Factor structure:
#   SelfDetermination = all Competence and Autonomy items (C1-C7, A1-A7)
#   Relatedness       = all HHR and HRR items
#
# No higher-order factor. Tests whether a broad two-dimensional structure
# fits as well as the theoretically specified four-factor model.
# -----------------------------------------------------------------------------

model4 <- '
  SelfDetermination =~ C1 + C2 + C3 + C4 + C5 + C6 + C7 +
                       A1 + A2 + A3 + A5 + A6 + A7
  Relatedness       =~ HHR1 + HHR2 + HHR3 + HHR4 +
                       HRR1 + HRR2 + HRR4
'

# -----------------------------------------------------------------------------
# MODEL 5: Correlated four-factor model (no higher-order factor).
# Motivated by the weak WellBeing -> HRR loading in Model 1 (0.571 vs
# 0.815-0.919 for other subscales) and HRR's lower inter-factor correlations.
#
# All four first-order factors (Competence, Autonomy, HHR, HRR) are specified
# with the same item assignments as Model 1, but allowed to correlate freely
# with each other. No higher-order WellBeing factor is imposed.
#
# This resolves the identification problems encountered with higher-order
# specifications by removing the structural constraints entirely.
#
# Comparison with Model 1: if Model 5 fits substantially better, it suggests
# the higher-order WB structure is not supported — the four subscales may share
# variance but do not converge on a single superordinate construct.
# Comparison with Model 4: tests whether four correlated factors fit better
# than two, i.e. whether the four-subscale granularity adds value.
#
# The HRR correlations in the output directly show whether HRR is weakly
# related to the other factors, without needing a higher-order specification.
# -----------------------------------------------------------------------------

model5 <- '
  # Four correlated first-order factors (same items as Model 1)
  # No higher-order factor — all inter-factor covariances are freely estimated
  Competence =~ C1 + C2 + C3 + C4 + C5 + C6 + C7
  Autonomy   =~ A1 + A2 + A3 + A5 + A6 + A7
  HHR        =~ HHR1 + HHR2 + HHR3 + HHR4
  HRR        =~ HRR1 + HRR2 + HRR4
'

# =============================================================================
# 3. MODEL ESTIMATION
# =============================================================================

cat(strrep("=", 60), "\n")
cat("ESTIMATING MODELS\n")
cat(strrep("=", 60), "\n\n")

estimate_model <- function(model_syntax, model_name, data,
                           optim.method   = "nlminb",
                           check.gradient = TRUE) {
  cat(sprintf("Estimating %s...\n", model_name))
  fit <- tryCatch(
    cfa(
      model_syntax,
      data           = data,
      estimator      = "WLSMV",
      ordered        = TRUE,
      std.lv         = FALSE,
      optim.method   = optim.method,
      check.gradient = check.gradient
    ),
    error = function(e) {
      cat(sprintf("  ERROR in %s: %s\n", model_name, e$message))
      return(NULL)
    }
  )
  if (!is.null(fit)) cat(sprintf("  %s estimated successfully.\n\n", model_name))
  return(fit)
}

fit1 <- estimate_model(model1, "Model 1 (Theoretical)", df_ord)
fit2 <- estimate_model(model2, "Model 2 (Empirical EFA)", df_ord)

# BFGS is more stable for bifactor models; check.gradient = FALSE prevents
# lavaan from refusing fit measures when gradient is near (but not exactly) zero.
# Heywood check below verifies whether the solution is genuinely valid.
fit3 <- estimate_model(
  model3, "Model 3 (Bifactor)", df_ord,
  optim.method   = "BFGS",
  check.gradient = FALSE
)

fit4 <- estimate_model(model4, "Model 4 (Two-Factor)", df_ord)
fit5 <- estimate_model(model5, "Model 5 (Correlated Four-Factor)", df_ord)

# Heywood check: negative residual (theta) or latent (psi) variances
# indicate impossible parameter estimates and genuine non-convergence.
if (!is.null(fit3)) {
  est      <- lavInspect(fit3, "est")
  neg_res  <- diag(est$theta)[diag(est$theta) < 0]
  neg_lv   <- diag(est$psi)[diag(est$psi) < 0]

  if (length(neg_res) > 0 || length(neg_lv) > 0) {
    cat("WARNING: Heywood case detected in Model 3 — genuine non-convergence.\n")
    if (length(neg_res) > 0) { cat("  Negative residual variances:\n"); print(neg_res) }
    if (length(neg_lv)  > 0) { cat("  Negative latent variances:\n");   print(neg_lv)  }
    cat("Model 3 excluded. The bifactor model is not identified with this data.\n\n")
    fit3 <- NULL
  } else {
    cat("Model 3 Heywood check passed.\n\n")
  }
}


# =============================================================================
# 4. FIT INDICES EXTRACTION
# =============================================================================

# AIC/BIC unavailable with WLSMV (weighted least squares, not ML).
# Robust scaled chi-square, CFI, TLI, RMSEA, and SRMR are reported.

extract_fit <- function(fit, model_name) {
  if (is.null(fit)) return(NULL)

  idx <- fitMeasures(fit, c(
    "chisq.scaled", "df.scaled", "pvalue.scaled",
    "cfi.robust", "tli.robust",
    "rmsea.robust", "rmsea.ci.lower.robust", "rmsea.ci.upper.robust",
    "srmr"
  ))

  tibble(
    Model       = model_name,
    chi2        = round(idx["chisq.scaled"], 3),
    df          = idx["df.scaled"],
    p_chi2      = round(idx["pvalue.scaled"], 4),
    CFI         = round(idx["cfi.robust"], 3),
    TLI         = round(idx["tli.robust"], 3),
    RMSEA       = round(idx["rmsea.robust"], 3),
    RMSEA_lower = round(idx["rmsea.ci.lower.robust"], 3),
    RMSEA_upper = round(idx["rmsea.ci.upper.robust"], 3),
    SRMR        = round(idx["srmr"], 3)
  )
}

fit_table <- bind_rows(
  extract_fit(fit1, "Model 1: Theoretical"),
  extract_fit(fit2, "Model 2: Empirical (EFA)"),
  extract_fit(fit3, "Model 3: Bifactor"),
  extract_fit(fit4, "Model 4: Two-Factor"),
  extract_fit(fit5, "Model 5: Correlated Four-Factor")
)

cat(strrep("=", 60), "\n")
cat("FIT INDICES SUMMARY\n")
cat(strrep("=", 60), "\n")
print(as.data.frame(fit_table), row.names = FALSE)
cat("\n")
cat("Fit thresholds (Hu & Bentler, 1999; Kline, 2016):\n")
cat("  CFI / TLI  >= 0.95 = good,  >= 0.90 = acceptable\n")
cat("  RMSEA      <= 0.05 = good,  <= 0.08 = acceptable,  > 0.10 = poor\n")
cat("  SRMR       <= 0.08 = good\n\n")


# =============================================================================
# 5. STANDARDISED FACTOR LOADINGS
# =============================================================================

extract_loadings <- function(fit, model_name) {
  if (is.null(fit)) return(NULL)

  standardizedSolution(fit) %>%
    filter(op == "=~") %>%
    select(
      Factor      = lhs,
      Item        = rhs,
      Std_Loading = est.std,
      SE          = se,
      Z           = z,
      p_value     = pvalue,
      CI_lower    = ci.lower,
      CI_upper    = ci.upper
    ) %>%
    mutate(
      across(where(is.numeric), ~ round(., 4)),
      Model = model_name,
      Sig   = case_when(
        p_value < .001 ~ "***",
        p_value < .01  ~ "**",
        p_value < .05  ~ "*",
        TRUE           ~ "n.s."
      )
    )
}

loadings1 <- extract_loadings(fit1, "Model 1: Theoretical")
loadings2 <- extract_loadings(fit2, "Model 2: Empirical")
loadings3 <- extract_loadings(fit3, "Model 3: Bifactor")
loadings4 <- extract_loadings(fit4, "Model 4: Two-Factor")
loadings5 <- extract_loadings(fit5, "Model 5: Correlated Four-Factor")

for (pair in list(
  list(loadings1, "MODEL 1 (Theoretical)"),
  list(loadings2, "MODEL 2 (Empirical EFA)"),
  list(loadings3, "MODEL 3 (Bifactor)"),
  list(loadings4, "MODEL 4 (Two-Factor)"),
  list(loadings5, "MODEL 5 (Correlated Four-Factor)")
)) {
  cat(strrep("=", 60), "\n")
  cat(sprintf("STANDARDISED FACTOR LOADINGS - %s\n", pair[[2]]))
  cat(strrep("=", 60), "\n")
  if (!is.null(pair[[1]])) {
    print(as.data.frame(pair[[1]] %>% select(-Model)), row.names = FALSE)
  } else {
    cat("  Not available (model did not converge or was excluded).\n")
  }
  cat("\n")
}


# =============================================================================
# 6. MODEL COMPARISON
# =============================================================================

cat(strrep("=", 60), "\n")
cat("MODEL COMPARISON\n")
cat(strrep("=", 60), "\n\n")

# AIC/BIC unavailable with WLSMV. All model pairs are non-nested:
# Models differ in item-factor assignments or presence of higher-order factor.
# Comparison is by direct fit index inspection only.
#
# Model 1 vs Model 5: same item-factor assignments, but Model 1 adds a
# higher-order WB factor constraining inter-factor covariances. These are
# nested (Model 1 is more constrained), so a difference test is valid.
cat("Note: AIC/BIC unavailable with WLSMV.\n")
cat("Models 1 and 5 are nested (Model 1 adds higher-order WB factor\n")
cat("to Model 5's correlated four-factor structure).\n")
cat("All other pairs are non-nested; comparison by fit indices only.\n\n")

if (!is.null(fit_table) && nrow(fit_table) > 0) {
  cat("Fit index comparison (better = higher CFI/TLI, lower RMSEA/SRMR):\n\n")
  print(
    as.data.frame(
      fit_table %>% select(Model, chi2, df, CFI, TLI, RMSEA,
                           RMSEA_lower, RMSEA_upper, SRMR)
    ),
    row.names = FALSE
  )
  cat("\n")
}

# Chi-square difference test: Model 5 vs Model 1 (nested)
# Model 1 imposes a higher-order WB factor on Model 5's correlated structure,
# reducing df. If Model 1 fits significantly worse, the higher-order structure
# is not supported by the data.
cat("Chi-square difference test: Model 5 (correlated) vs Model 1 (higher-order):\n")
cat("Significant result (p < .05) = higher-order WB structure significantly\n")
cat("worsens fit, suggesting the four factors do not converge on a single construct.\n\n")

if (!is.null(fit5) && !is.null(fit1)) {
  tryCatch({
    # Model 5 is less constrained (more df), Model 1 is more constrained (fewer df)
    # lavTestLRT expects models ordered from least to most constrained
    diff_result <- lavTestLRT(fit5, fit1)
    print(diff_result)
  }, error = function(e) {
    cat(sprintf("Chi-square difference test failed: %s\n", e$message))
    cat("Compare models via fit indices directly.\n\n")
  })
}


# =============================================================================
# 7. MODIFICATION INDICES (Model 1)
# =============================================================================

cat("\n")
cat(strrep("=", 60), "\n")
cat("MODIFICATION INDICES - MODEL 1 (top 15, sorted by MI)\n")
cat(strrep("=", 60), "\n\n")

if (!is.null(fit1)) {
  mi <- modindices(fit1, sort. = TRUE, maximum.number = 15)
  print(
    as.data.frame(
      mi %>%
        select(lhs, op, rhs, mi, epc, sepc.all) %>%
        mutate(across(where(is.numeric), ~ round(., 3)))
    ),
    row.names = FALSE
  )
  cat("\nmi       = modification index (chi-square improvement if path freed)\n")
  cat("epc      = expected parameter change\n")
  cat("sepc.all = standardised expected parameter change\n")
  cat("Note: only free parameters with clear theoretical justification.\n\n")
}


# =============================================================================
# 8. OMEGA HIERARCHICAL
# =============================================================================

cat(strrep("=", 60), "\n")
cat("OMEGA HIERARCHICAL FROM BIFACTOR MODEL (Model 3)\n")
cat(strrep("=", 60), "\n\n")

if (is.null(fit3)) {
  cat("Model 3 did not converge — omega hierarchical cannot be computed from CFA.\n")
  cat("The Schmid-Leiman estimate from the reliability section (omega_h = 0.614)\n")
  cat("remains the best available estimate. Report this limitation explicitly.\n\n")
} else {
  tryCatch({
    rel <- reliability(fit3)
    cat("Reliability estimates from bifactor model:\n")
    print(round(rel, 4))
    cat("\nomega.h = proportion of total score variance attributable to the\n")
    cat("general WB factor after removing group factor variance.\n\n")
  }, error = function(e) {
    cat(sprintf("Omega hierarchical computation failed: %s\n", e$message))
  })
}


# =============================================================================
# 9. MODEL-IMPLIED FACTOR CORRELATIONS
# =============================================================================

extract_implied_factor_corr <- function(fit, model_name, factor_names) {
  if (is.null(fit)) return(NULL)
  tryCatch({
    psi      <- lavInspect(fit, "cor.lv")
    fo_names <- factor_names[factor_names %in% rownames(psi)]
    psi_fo   <- psi[fo_names, fo_names]
    as.data.frame(round(psi_fo, 4)) %>%
      rownames_to_column("Factor") %>%
      mutate(Model = model_name)
  }, error = function(e) {
    cat(sprintf("Could not extract factor correlations for %s: %s\n",
                model_name, e$message))
    return(NULL)
  })
}

cat(strrep("=", 60), "\n")
cat("MODEL-IMPLIED FACTOR CORRELATIONS\n")
cat(strrep("=", 60), "\n\n")

corr1 <- extract_implied_factor_corr(
  fit1, "Model 1: Theoretical",
  c("Competence", "Autonomy", "HHR", "HRR")
)
corr2 <- extract_implied_factor_corr(
  fit2, "Model 2: Empirical",
  c("F1_SelfEfficacy", "F2_SocialRelated", "F3_Evaluative", "F4_ExternalReg")
)
corr4 <- extract_implied_factor_corr(
  fit4, "Model 4: Two-Factor",
  c("SelfDetermination", "Relatedness")
)
corr5 <- extract_implied_factor_corr(
  fit5, "Model 5: Correlated Four-Factor",
  c("Competence", "Autonomy", "HHR", "HRR")
)

if (!is.null(corr1)) {
  cat("Model 1:\n")
  print(as.data.frame(corr1 %>% select(-Model)), row.names = FALSE)
  cat("\n")
}
if (!is.null(corr2)) {
  cat("Model 2:\n")
  print(as.data.frame(corr2 %>% select(-Model)), row.names = FALSE)
  cat("\n")
}
if (!is.null(corr4)) {
  cat("Model 4:\n")
  print(as.data.frame(corr4 %>% select(-Model)), row.names = FALSE)
  cat("\n")
}
if (!is.null(corr5)) {
  cat("Model 5:\n")
  print(as.data.frame(corr5 %>% select(-Model)), row.names = FALSE)
  cat("\n")
}


# =============================================================================
# 10. FACTOR LOADING PLOTS
# =============================================================================

plot_loadings <- function(loadings_df, model_name, output_path, item_list) {
  if (is.null(loadings_df)) return(invisible(NULL))

  plot_data <- loadings_df %>%
    filter(Item %in% item_list) %>%
    mutate(
      Significant = p_value < .05,
      Item        = factor(Item, levels = rev(item_list))
    )

  p <- ggplot(plot_data, aes(x = Std_Loading, y = Item, colour = Factor)) +
    geom_vline(xintercept = 0.30, linetype = "dashed",
               colour = "#cccccc", linewidth = 0.5) +
    geom_vline(xintercept = 0.50, linetype = "dashed",
               colour = "#999999", linewidth = 0.5) +
    geom_vline(xintercept = 0, colour = "black", linewidth = 0.3) +
    # geom_errorbar with orientation = "y" replaces deprecated geom_errorbarh
    geom_errorbar(
      aes(xmin = CI_lower, xmax = CI_upper, ymin = Item, ymax = Item),
      orientation = "y", linewidth = 0.5, alpha = 0.6, width = 0.3
    ) +
    geom_point(aes(shape = Significant), size = 3) +
    scale_shape_manual(
      values = c("TRUE" = 16, "FALSE" = 1),
      labels = c("TRUE" = "p < .05", "FALSE" = "n.s.")
    ) +
    scale_x_continuous(limits = c(-0.2, 1.2), breaks = seq(0, 1, 0.2)) +
    labs(
      title    = paste("Standardised Factor Loadings -", model_name),
      subtitle = "Dashed lines: 0.30 (acceptable) and 0.50 (good) thresholds",
      x        = "Standardised Loading",
      y        = NULL,
      colour   = "Factor",
      shape    = "Significance"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      legend.position  = "right",
      panel.grid.minor = element_blank(),
      plot.title       = element_text(face = "bold")
    )

  ggsave(output_path, plot = p, width = 9, height = 7, dpi = 300)
  cat(sprintf("Loading plot saved to %s\n", output_path))
}


# =============================================================================
# 11. SAVE ALL OUTPUTS
# =============================================================================

output_dir <- "output/cfa"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

write_csv(fit_table, file.path(output_dir, "fit_indices.csv"))
cat(sprintf("Fit indices saved to %s/fit_indices.csv\n", output_dir))

if (!is.null(loadings1))
  write_csv(loadings1, file.path(output_dir, "loadings_model1_theoretical.csv"))
if (!is.null(loadings2))
  write_csv(loadings2, file.path(output_dir, "loadings_model2_empirical.csv"))
if (!is.null(loadings3))
  write_csv(loadings3, file.path(output_dir, "loadings_model3_bifactor.csv"))

if (!is.null(corr1))
  write_csv(corr1, file.path(output_dir, "factor_correlations_model1.csv"))
if (!is.null(corr2))
  write_csv(corr2, file.path(output_dir, "factor_correlations_model2.csv"))

plot_loadings(loadings1, "Model 1: Theoretical",
              file.path(output_dir, "loadings_plot_model1.png"), items)
plot_loadings(loadings2, "Model 2: Empirical EFA",
              file.path(output_dir, "loadings_plot_model2.png"), items)
plot_loadings(loadings4, "Model 4: Two-Factor",
              file.path(output_dir, "loadings_plot_model4.png"), items)
plot_loadings(loadings5, "Model 5: Correlated Four-Factor",
              file.path(output_dir, "loadings_plot_model5.png"), items)

if (!is.null(loadings4))
  write_csv(loadings4, file.path(output_dir, "loadings_model4_twofactor.csv"))
if (!is.null(loadings5))
  write_csv(loadings5, file.path(output_dir, "loadings_model5_hrr_independent.csv"))

if (!is.null(corr4))
  write_csv(corr4, file.path(output_dir, "factor_correlations_model4.csv"))
if (!is.null(corr5))
  write_csv(corr5, file.path(output_dir, "factor_correlations_model5.csv"))

for (pair in list(
  list(fit1, "summary_model1.txt"),
  list(fit2, "summary_model2.txt"),
  list(fit3, "summary_model3.txt"),
  list(fit4, "summary_model4.txt"),
  list(fit5, "summary_model5.txt")
)) {
  if (!is.null(pair[[1]])) {
    sink(file.path(output_dir, pair[[2]]))
    summary(pair[[1]], fit.measures = TRUE, standardized = TRUE, rsquare = TRUE)
    sink()
  }
}

cat("\n")
cat(strrep("=", 60), "\n")
cat(sprintf("All CFA outputs saved to '%s/'\n", output_dir))
cat(strrep("=", 60), "\n")
cat("CFA analysis complete.\n")