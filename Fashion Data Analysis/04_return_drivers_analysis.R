# 04_return_drivers_analysis.R
# Purpose: Understand drivers of product returns (statistical tests, effect sizes, feature importance)
# Inputs: cleaned_fashion_boutique.csv
# Outputs: outputs_return_drivers/ (plots, CSVs), logistic regression summary, RF importance, ROC
# Author: [Your Name]
# Date: [YYYY-MM-DD]



# -----------------------------
# 0) Setup
# -----------------------------
packages <- c(
  "tidyverse", "lubridate", "scales", "broom", "broom.mixed", "data.table",
  "rcompanion", "effsize", "randomForest", "caret", "pROC", "vip", "patchwork"
)
for (p in packages) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)

library(tidyverse)
library(lubridate)
library(scales)
library(broom)
library(data.table)
library(rcompanion)   # cramerV
library(effsize)      # cohen.d
library(randomForest)
library(caret)
library(pROC)
library(vip)
library(patchwork)






# -----------------------------
# Paths & output folder
# -----------------------------
CLEANED_DATA_PATH <- "cleaned_fashion_boutique.csv"
OUT_DIR <- "outputs_return_drivers"
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

save_plot <- function(plot, filename, width = 9, height = 6) {
  ggsave(filename, plot, width = width, height = height, dpi = 300)
}





df <- readr::read_csv("cleaned_fashion_boutique.csv")




cat("\nRows:", nrow(df), " Columns:", ncol(df), "\n")
cat("Return rate:", mean(df$is_returned, na.rm = TRUE) %>% scales::percent(accuracy = 0.1), "\n\n")

# Create convenience response factor for modeling
df <- df %>%
  mutate(returned_factor = factor(if_else(is_returned, "Returned", "Not_Returned")))

# -----------------------------
# 2) Exploratory plots (overview)
# -----------------------------
# Return rate by category
plot_cat <- df %>%
  group_by(category) %>%
  summarise(units = n(), return_rate = mean(is_returned, na.rm = TRUE)) %>%
  arrange(desc(return_rate)) %>%
  ggplot(aes(x = reorder(category, return_rate), y = return_rate, fill = category)) +
  geom_col(show.legend = FALSE) +
  scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(title = "Return Rate by Category", x = "Category", y = "Return Rate") +
  theme_minimal()
save_plot(plot_cat, file.path(OUT_DIR, "return_rate_by_category.png"))

# Return rate by brand (top 15 brands by units to avoid overcrowding)
top_brands <- df %>% count(brand, sort = TRUE) %>% slice_head(n = 15) %>% pull(brand)
plot_brand <- df %>%
  filter(brand %in% top_brands) %>%
  group_by(brand) %>%
  summarise(units = n(), return_rate = mean(is_returned, na.rm = TRUE)) %>%
  ggplot(aes(x = reorder(brand, return_rate), y = return_rate, fill = brand)) +
  geom_col(show.legend = FALSE) +
  scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(title = "Return Rate by Brand (Top 15 by volume)", x = "Brand", y = "Return Rate") +
  theme_minimal()
save_plot(plot_brand, file.path(OUT_DIR, "return_rate_by_brand_top15.png"))

# Return reasons distribution (for returned items)
plot_reasons <- df %>%
  filter(is_returned == TRUE) %>%
  count(return_reason, sort = TRUE) %>%
  mutate(pct = n / sum(n)) %>%
  ggplot(aes(x = reorder(return_reason, pct), y = pct, fill = return_reason)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Return Reasons (Returned Items Only)", x = "Return Reason", y = "Share of Returns") +
  theme_minimal()
save_plot(plot_reasons, file.path(OUT_DIR, "return_reasons.png"))

# Boxplot: current_price vs returned
plot_price <- ggplot(df, aes(x = returned_factor, y = current_price)) +
  geom_boxplot() +
  scale_y_continuous(labels = dollar_format()) +
  labs(title = "Price Distribution by Return Status", x = "Return Status", y = "Current Price") +
  theme_minimal()
save_plot(plot_price, file.path(OUT_DIR, "price_boxplot_by_return.png"))

# Density: customer_rating by return status
plot_rating <- ggplot(df, aes(x = customer_rating, color = returned_factor, fill = returned_factor)) +
  geom_density(alpha = 0.2) +
  labs(title = "Customer Rating Distribution by Return Status", x = "Customer Rating", y = "Density") +
  theme_minimal()
save_plot(plot_rating, file.path(OUT_DIR, "rating_density_by_return.png"))

# Discounted vs return rate
plot_disc <- df %>%
  group_by(discounted) %>%
  summarise(return_rate = mean(is_returned, na.rm = TRUE), n = n()) %>%
  ggplot(aes(x = as.factor(discounted), y = return_rate, fill = as.factor(discounted))) +
  geom_col(show.legend = FALSE) +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Return Rate: Discounted vs Not Discounted", x = "Discounted", y = "Return Rate") +
  theme_minimal()
save_plot(plot_disc, file.path(OUT_DIR, "discounted_vs_return_rate.png"))

# -----------------------------
# 3) Univariate statistical tests & effect sizes
#    - Categorical: chi-square + Cramer's V
#    - Numeric: t-test/Wilcoxon + Cohen's d
# -----------------------------
# Helper to compute Cramer's V safely
cramers_v_safe <- function(tbl) {
  # tbl: contingency table
  # rcompanion::cramerV handles zero entries
  rcompanion::cramerV(tbl, bias.correct = TRUE)
}

# Categorical variables to test
cat_vars <- c("category", "brand", "season", "size", "color", "discounted", "price_discrepancy", "rating_imputed")

cat_tests <- list()
for (v in cat_vars) {
  # make 2-way table returned x v
  if (!v %in% names(df)) next
  tbl <- table(df[[v]], df$is_returned)
  # require at least 2 levels in variable
  if (nrow(tbl) < 2) next
  chi <- tryCatch(chisq.test(tbl), error = function(e) NULL)
  cv <- tryCatch(cramers_v_safe(tbl), error = function(e) NA_real_)
  chi_p <- if (!is.null(chi)) chi$p.value else NA_real_
  cat_tests[[v]] <- tibble(
    variable = v,
    chi_square_p = chi_p,
    cramers_v = as.numeric(cv)
  )
}
cat_tests_df <- bind_rows(cat_tests)
readr::write_csv(cat_tests_df, file.path(OUT_DIR, "categorical_tests_cramersV.csv"))
cat("Categorical tests saved.\n")

# Numeric variables to test
num_vars <- c("original_price", "markdown_percentage", "current_price", "stock_quantity", "customer_rating", "discount_amount", "profit_margin")
num_tests <- list()
for (v in num_vars) {
  if (!v %in% names(df)) next
  x_ret <- df %>% filter(is_returned == TRUE) %>% pull(.data[[v]])
  x_not <- df %>% filter(is_returned == FALSE) %>% pull(.data[[v]])
  # remove NAs
  x_ret <- x_ret[!is.na(x_ret)]
  x_not <- x_not[!is.na(x_not)]
  if (length(x_ret) < 3 || length(x_not) < 3) next
  # Use t-test if approx normal; otherwise use wilcox.test (we'll compute both)
  t_res <- tryCatch(t.test(x_ret, x_not), error = function(e) NULL)
  wilc <- tryCatch(wilcox.test(x_ret, x_not), error = function(e) NULL)
  # effect size: Cohen's d
  cohen <- tryCatch(effsize::cohen.d(x_ret, x_not, na.rm = TRUE)$estimate, error = function(e) NA_real_)
  num_tests[[v]] <- tibble(
    variable = v,
    t_p_value = if (!is.null(t_res)) t_res$p.value else NA_real_,
    wilcox_p_value = if (!is.null(wilc)) wilc$p.value else NA_real_,
    cohen_d = as.numeric(cohen)
  )
}
num_tests_df <- bind_rows(num_tests)
readr::write_csv(num_tests_df, file.path(OUT_DIR, "numeric_tests_cohend.csv"))
cat("Numeric tests saved.\n")

# -----------------------------
# 4) Multivariate logistic regression (odds ratios + CIs)
#    - Build a parsimonious model with interpretable variables
# -----------------------------
# Select predictors (avoid too many sparse factor levels; group rare levels)
# Strategy: keep category, top N brands (others grouped as 'Other'), season, size, color (maybe top colors),
# customer_rating, current_price, discounted, markdown_percentage, stock_quantity, price_discrepancy

# Helper to group rare levels into "Other"
group_top_levels <- function(x, n = 10) {
  x <- as.character(x)
  topn <- names(sort(table(x), decreasing = TRUE))[1:min(n, length(unique(x)))]
  f <- ifelse(x %in% topn, x, "Other")
  factor(f)
}

df_model <- df %>%
  mutate(
    brand_top = group_top_levels(brand, n = 10),
    color_top = group_top_levels(color, n = 8),
    size = fct_lump(size, n = 6, other_level = "Other"),
    category = fct_lump(category, n = 8, other_level = "Other"),
    season = fct_explicit_na(season, na_level = "Unknown"),
    returned = if_else(is_returned, 1L, 0L)
  ) %>%
  select(returned, returned_factor, category, brand_top, season, size, color_top,
         original_price, markdown_percentage, current_price, stock_quantity,
         customer_rating, discounted, price_discrepancy, discount_amount, profit_margin) %>%
  mutate_if(is.logical, ~ ifelse(is.na(.), FALSE, .))

# Fit logistic regression (use glm)
# Use formula with main effects only for interpretability
formula_glm <- as.formula("returned ~ category + brand_top + season + size + color_top + current_price + customer_rating + discounted + price_discrepancy + discount_amount + profit_margin + stock_quantity")

glm_fit <- glm(formula_glm, data = df_model, family = binomial(link = "logit"))
summary(glm_fit)

# Extract coefficients -> odds ratios
coefs <- broom::tidy(glm_fit, conf.int = TRUE, exponentiate = TRUE)
coefs <- coefs %>%
  rename(odds_ratio = estimate, conf_low = conf.low, conf_high = conf.high) %>%
  arrange(desc(abs(odds_ratio - 1)))

readr::write_csv(coefs, file.path(OUT_DIR, "logistic_odds_ratios.csv"))

# Plot top variables by OR magnitude (exclude intercept)
plot_or <- coefs %>%
  filter(term != "(Intercept)") %>%
  mutate(term = factor(term)) %>%  # ensure term is a factor
  mutate(term = fct_reorder(term, odds_ratio)) %>%
  top_n(30, wt = abs(odds_ratio - 1)) %>%
  ggplot(aes(x = term, y = odds_ratio)) +
  geom_point(color = "steelblue", size = 3) +
  geom_errorbar(aes(ymin = conf_low, ymax = conf_high), width = 0.2, color = "gray40") +
  coord_flip() +
  scale_y_log10() +
  labs(
    title = "Top Logistic Regression Odds Ratios (log scale)",
    x = "Predictor",
    y = "Odds Ratio (log scale)"
  ) +
  theme_minimal()

# Check model diagnostics: pseudo-R2, AIC
library(MASS)
pR2_val <- 1 - glm_fit$deviance / glm_fit$null.deviance
glm_aic <- AIC(glm_fit)
cat("Pseudo-R2 (1 - deviance/null):", round(pR2_val, 4), "\nAIC:", glm_aic, "\n")

# -----------------------------
# 5) Random Forest (feature importance, partial dependence)
# -----------------------------
library(dplyr)

rf_df <- df_model %>%
  # Make sure returned_factor exists (overwrite safely)
  mutate(returned_factor = factor(if_else(returned == 1, "Returned", "Not_Returned"))) %>%
  dplyr::select(-returned) %>%  # drop numeric version
  mutate(across(where(is.character), as.factor))

colnames(df_model)

# Split into train/test (70/30 stratified)
set.seed(123)
train_idx <- createDataPartition(rf_df$returned_factor, p = 0.7, list = FALSE)
train_rf <- rf_df[train_idx, ]
test_rf  <- rf_df[-train_idx, ]

# Train random forest (caret wrapper)
rf_control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
# Simplify formula to avoid too many levels in tree training
rf_formula <- as.formula("returned_factor ~ .")
set.seed(123)
rf_fit <- train(rf_formula, data = train_rf, method = "rf", metric = "ROC", trControl = rf_control, importance = TRUE, ntree = 500)

# Variable importance
vi <- vip::vip(rf_fit$finalModel, num_features = 25)
# Save vip plot
png(file.path(OUT_DIR, "rf_variable_importance.png"), width = 1200, height = 800, res = 150)
vip::vip(rf_fit$finalModel, num_features = 25)
dev.off()

# Predict on test set and compute ROC/AUC
rf_preds <- predict(rf_fit, test_rf, type = "prob")
roc_rf <- pROC::roc(response = test_rf$returned_factor, predictor = rf_preds[, "Returned"], levels = rev(levels(test_rf$returned_factor)))
auc_rf <- pROC::auc(roc_rf)
cat("Random Forest AUC:", round(as.numeric(auc_rf), 4), "\n")

# Save ROC plot
png(file.path(OUT_DIR, "rf_roc_test.png"), width = 800, height = 600, res = 150)
plot(roc_rf, main = paste0("Random Forest ROC (AUC = ", round(as.numeric(auc_rf), 3), ")"))
dev.off()

# Save RF predictions with key info for analysis
test_out <- test_rf %>%
  mutate(prob_returned = rf_preds[, "Returned"],
         predicted_class = predict(rf_fit, test_rf))
readr::write_csv(test_out %>% mutate(across(everything(), as.character)), file.path(OUT_DIR, "rf_test_predictions.csv"))






# -----------------------------
# 6) Summarize & Export "significant drivers"
#    - Combine results from univariate tests, ORs, and RF importance
# -----------------------------
library(caret)
library(dplyr)  # <-- add this

set.seed(123)
rf_fit <- train(
  returned_factor ~ .,
  data = rf_df,
  method = "rf",
  importance = TRUE,
  ntree = 500
)

# Extract variable importance from RF
rf_imp_raw <- varImp(rf_fit)$importance %>%
  rownames_to_column("feature")

# If multiple class columns exist, compute mean importance
rf_imp_df <- rf_imp_raw %>%
  mutate(mean_imp = rowMeans(dplyr::select(., -feature), na.rm = TRUE)) %>%
  arrange(desc(mean_imp)) %>%
  mutate(rank = row_number())


# Save RF importance table
readr::write_csv(rf_imp_df, file.path(OUT_DIR, "rf_variable_importance_table.csv"))


# -----------------------------------
# Combine results across methods
# -----------------------------------

# Logistic regression significant terms (p < 0.05)
glm_sig <- glm_tidy_full %>%
  filter(!is.na(p.value) & p.value < 0.05) %>%
  dplyr::select(term, estimate, p.value)


# Categorical: Cramer's V > 0.1
cat_sig_cats <- cat_tests_df %>%
  filter(!is.na(cramers_v) & cramers_v > 0.1) %>%
  pull(variable)

# Numeric: Cohen's d > 0.2
num_sig <- num_tests_df %>%
  filter(!is.na(cohen_d) & abs(cohen_d) > 0.2) %>%
  pull(variable)

# RF: top 15 features
rf_top_feats <- rf_imp_df %>%
  slice_head(n = 15) %>%
  pull(feature)

# Bundle drivers into list
drivers <- list(
  categorical_cramersv = cat_sig_cats,
  numeric_cohend = num_sig,
  glm_sig_terms = glm_sig$term,
  rf_top = rf_top_feats
)

# Save combined summary
readr::write_csv(
  tibble(
    driver_type = names(drivers),
    drivers = sapply(drivers, function(x) paste(x, collapse = "; "))
  ),
  file.path(OUT_DIR, "combined_drivers_summary.csv")
)






















# -----------------------------
# 7) Visualize top drivers (combine plots)
# -----------------------------
# Example: plot return_rate by top RF features if categorical; for numeric show boxplots
for (feat in top_features) {
  if (!feat %in% names(train_rf)) next
  
  if (is.factor(train_rf[[feat]]) || is.character(train_rf[[feat]])) {
    p <- df %>% 
      group_by(.data[[feat]]) %>%
      summarise(return_rate = mean(is_returned, na.rm = TRUE), 
                n = n(), .groups = "drop") %>%
      arrange(desc(return_rate)) %>%
      ggplot(aes(x = reorder(as.character(.data[[feat]]), return_rate), 
                 y = return_rate)) +
      geom_col() +
      coord_flip() +
      scale_y_continuous(labels = scales::percent_format()) +
      labs(title = paste("Return rate by", feat), x = feat, y = "Return rate") +
      theme_minimal()
  } else {
    p <- ggplot(df, aes(x = returned_factor, y = .data[[feat]])) +
      geom_boxplot() +
      labs(title = paste(feat, "by return status"), 
           x = "Return status", y = feat) +
      theme_minimal()
  }
  
  plots_list[[feat]] <- p
}


# Arrange and save combined plot
nplots <- length(plots_list)
if (nplots > 0) {
  # combine first 6 into a grid
  plt_combined <- wrap_plots(plots_list[1:min(6, nplots)], ncol = 2)
  save_plot(plt_combined, file.path(OUT_DIR, "top_driver_plots_combined.png"), width = 12, height = 8)
}

# -----------------------------
# 8) Final notes: outputs & files
# -----------------------------
cat("\nAll outputs saved to:", OUT_DIR, "\n")
cat("Files written (examples):\n")
print(list.files(OUT_DIR, full.names = TRUE) %>% head(50))

# End of script


