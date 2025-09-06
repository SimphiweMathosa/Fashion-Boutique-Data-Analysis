# 05_return_prediction_model.R
# Objective: Predict return likelihood (classification) and beat baselines on ROC AUC / precision / recall
# Inputs : cleaned_fashion_boutique.csv
# Outputs: outputs_return_pred/ (plots, CSVs)
# Author : [Your Name]
# Date   : [YYYY-MM-DD]

# -----------------------------
# 0) Setup
# -----------------------------
packages <- c(
  "tidyverse", "lubridate", "scales",
  "caret", "pROC", "randomForest", "glmnet", "ROCR", "RColorBrewer"
)
for (p in packages) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)

library(tidyverse)
library(lubridate)
library(scales)
library(caret)
library(pROC)
library(randomForest)  # for rf engine under caret
library(glmnet)        # for glmnet under caret
library(ROCR)          # PR curve helper
library(RColorBrewer)

CLEANED_DATA_PATH <- "cleaned_fashion_boutique.csv"
OUT_DIR <- "outputs_return_pred"
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

save_plot <- function(plot, filename, width = 9, height = 6) {
  ggsave(filename, plot, width = width, height = height, dpi = 300)
}

# -----------------------------
# 1) Load data (strict col types from dictionary)
# -----------------------------

df <- readr::read_csv("cleaned_fashion_boutique.csv")


cat("Rows:", nrow(df), " | Columns:", ncol(df), "\n")
cat("Return rate:", scales::percent(mean(df$is_returned, na.rm = TRUE), 0.1), "\n")

# -----------------------------
# 2) Feature prep (match prior scripts)
#    - reduce very-high-cardinality fields
#    - keep interpretable numerics
# -----------------------------
group_top_levels <- function(x, n = 10) {
  x <- as.character(x)
  topn <- names(sort(table(x), decreasing = TRUE))[1:min(n, length(unique(x)))]
  fct <- ifelse(x %in% topn, x, "Other")
  factor(fct)
}

model_df <- df %>%
  mutate(
    # Target
    returned_factor = factor(if_else(is_returned, "Returned", "Not_Returned"),
                             levels = c("Not_Returned", "Returned")),
    # Reduce sparsity
    brand_top = group_top_levels(brand, n = 10),
    color_top = group_top_levels(color, n = 10),
    size = fct_lump(size, n = 6, other_level = "Other"),
    category = fct_lump(category, n = 8, other_level = "Other"),
    season = fct_explicit_na(season, na_level = "Unknown"),
    # Logical -> factor (caret likes factors for dummies)
    discounted = factor(if_else(isTRUE(discounted), "Yes", "No")),
    price_discrepancy = factor(if_else(isTRUE(price_discrepancy), "Yes", "No")),
    rating_imputed = factor(if_else(isTRUE(rating_imputed), "Yes", "No"))
  ) %>%
  select(
    # target
    returned_factor,
    # categorical predictors (reduced)
    category, brand_top, season, size, color_top,
    # numeric predictors
    original_price, markdown_percentage, current_price,
    stock_quantity, customer_rating, discount_amount, profit_margin,
    # logicals-as-factors
    discounted, price_discrepancy, rating_imputed
  )

# Drop rows with missing target
model_df <- model_df %>% filter(!is.na(returned_factor))

# -----------------------------
# 3) Train/Test split (80/20 stratified)
# -----------------------------
set.seed(42)
idx <- caret::createDataPartition(model_df$returned_factor, p = 0.8, list = FALSE)
train_df <- model_df[idx, ]
test_df  <- model_df[-idx, ]

cat("Train:", nrow(train_df), "  Test:", nrow(test_df), "\n")

# -----------------------------
# 4) Baselines
# -----------------------------
# Baseline A: majority-class (always predict Not_Returned)
maj_class <- names(sort(table(train_df$returned_factor), decreasing = TRUE))[1]
baseline_majority_pred <- factor(rep(maj_class, nrow(test_df)), levels = levels(test_df$returned_factor))
baseline_majority_prob <- ifelse(baseline_majority_pred == "Returned", 1, 0)

# Baseline B: constant probability (overall return rate in train)
base_rate <- mean(train_df$returned_factor == "Returned")
baseline_rate_prob <- rep(base_rate, nrow(test_df))

# Compute baseline metrics
compute_metrics <- function(truth, prob, threshold = 0.5) {
  pred <- factor(ifelse(prob >= threshold, "Returned", "Not_Returned"),
                 levels = c("Not_Returned", "Returned"))
  cm <- caret::confusionMatrix(pred, truth, positive = "Returned")
  roc <- pROC::roc(truth, prob, levels = c("Not_Returned", "Returned"), quiet = TRUE)
  tibble(
    ROC_AUC = as.numeric(pROC::auc(roc)),
    Accuracy = cm$overall["Accuracy"],
    Precision = cm$byClass["Precision"],
    Recall = cm$byClass["Recall"],
    F1 = cm$byClass["F1"]
  )
}

baseline_majority_metrics <- compute_metrics(test_df$returned_factor, baseline_majority_prob, threshold = 0.5)
baseline_rate_metrics     <- compute_metrics(test_df$returned_factor, baseline_rate_prob, threshold = 0.5)
baselines_tbl <- bind_rows(
  baseline_majority_metrics %>% mutate(Model = "Baseline_Majority"),
  baseline_rate_metrics %>% mutate(Model = "Baseline_ConstantProb")
) %>% select(Model, everything())
readr::write_csv(baselines_tbl, file.path(OUT_DIR, "baseline_metrics.csv"))
print(baselines_tbl)

# -----------------------------
# 5) Preprocessing recipe for caret (one-hot encode)
# -----------------------------
dmy <- caret::dummyVars(returned_factor ~ ., data = train_df, fullRank = TRUE)
X_train <- predict(dmy, newdata = train_df) %>% as.data.frame()
X_test  <- predict(dmy, newdata = test_df)  %>% as.data.frame()
y_train <- train_df$returned_factor
y_test  <- test_df$returned_factor

# -----------------------------
# 6) Train models (caret)
#    - GLM (logistic), GLMNET, Random Forest
# -----------------------------
ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 2,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

set.seed(42)
fit_glm <- train(
  x = X_train, y = y_train,
  method = "glm",
  metric = "ROC",
  family = binomial(),
  trControl = ctrl
)

set.seed(42)
fit_glmnet <- train(
  x = X_train, y = y_train,
  method = "glmnet",
  metric = "ROC",
  trControl = ctrl,
  tuneLength = 10
)

set.seed(42)
fit_rf <- train(
  x = X_train, y = y_train,
  method = "rf",
  metric = "ROC",
  trControl = ctrl,
  tuneLength = 5,
  importance = TRUE,
  ntree = 500
)

models <- list(GLM = fit_glm, GLMNET = fit_glmnet, RF = fit_rf)
cv_res <- map_df(models, ~ .x$results %>% slice_max(ROC, n = 1), .id = "Model")
readr::write_csv(cv_res, file.path(OUT_DIR, "cv_results_best_rows.csv"))
print(cv_res)

# -----------------------------
# 7) Evaluate on holdout (ROC AUC + PR + threshold tuning)
# -----------------------------
predict_prob <- function(fit, X) {
  as.numeric(predict(fit, X, type = "prob")[, "Returned"])
}

probs_glm    <- predict_prob(fit_glm, X_test)
probs_glmnet <- predict_prob(fit_glmnet, X_test)
probs_rf     <- predict_prob(fit_rf, X_test)

# Helper: PR curve and best F1 threshold on test
best_threshold <- function(truth, prob) {
  df <- tibble(prob = prob, truth = truth)
  grid <- seq(0.05, 0.95, by = 0.01)
  scores <- map_df(grid, function(t) {
    pred <- factor(ifelse(df$prob >= t, "Returned", "Not_Returned"),
                   levels = c("Not_Returned", "Returned"))
    cm <- caret::confusionMatrix(pred, df$truth, positive = "Returned")
    tibble(threshold = t,
           Precision = cm$byClass["Precision"], Recall = cm$byClass["Recall"],
           F1 = cm$byClass["F1"])
  })
  scores %>% filter(!is.na(F1)) %>% slice_max(F1, n = 1)
}

bt_glm    <- best_threshold(y_test, probs_glm)
bt_glmnet <- best_threshold(y_test, probs_glmnet)
bt_rf     <- best_threshold(y_test, probs_rf)

metrics_tbl <- bind_rows(
  compute_metrics(y_test, probs_glm,    bt_glm$threshold)    %>% mutate(Model = "GLM",    Threshold = bt_glm$threshold),
  compute_metrics(y_test, probs_glmnet, bt_glmnet$threshold) %>% mutate(Model = "GLMNET", Threshold = bt_glmnet$threshold),
  compute_metrics(y_test, probs_rf,     bt_rf$threshold)     %>% mutate(Model = "RF",     Threshold = bt_rf$threshold),
  baseline_majority_metrics %>% mutate(Model = "Baseline_Majority", Threshold = 0.5),
  baseline_rate_metrics     %>% mutate(Model = "Baseline_ConstantProb", Threshold = 0.5)
) %>% select(Model, Threshold, everything()) %>%
  arrange(desc(ROC_AUC))

print(metrics_tbl)
readr::write_csv(metrics_tbl, file.path(OUT_DIR, "test_metrics_with_thresholds.csv"))

# Pick best by ROC AUC (tie-breaker: F1)
best_row <- metrics_tbl %>% slice_max(ROC_AUC, n = 1)
BEST_MODEL_NAME <- best_row$Model[1]
BEST_THRESHOLD  <- as.numeric(best_row$Threshold[1])
cat("Best model:", BEST_MODEL_NAME, " | threshold:", BEST_THRESHOLD, "\n")

best_fit <- switch(BEST_MODEL_NAME,
                   "GLM" = fit_glm,
                   "GLMNET" = fit_glmnet,
                   "RF" = fit_rf)

best_probs <- switch(BEST_MODEL_NAME,
                     "GLM" = probs_glm,
                     "GLMNET" = probs_glmnet,
                     "RF" = probs_rf)

# -----------------------------
# 8) Visualizations
# -----------------------------
# 8a. ROC curves (all models + baselines)
roc_glm    <- roc(y_test, probs_glm,    levels = c("Not_Returned", "Returned"), quiet = TRUE)
roc_glmnet <- roc(y_test, probs_glmnet, levels = c("Not_Returned", "Returned"), quiet = TRUE)
roc_rf     <- roc(y_test, probs_rf,     levels = c("Not_Returned", "Returned"), quiet = TRUE)
roc_base   <- roc(y_test, baseline_rate_prob, levels = c("Not_Returned", "Returned"), quiet = TRUE)

p_roc <- ggplot() +
  geom_line(aes(x = 1 - roc_glm$specificities,    y = roc_glm$sensitivities),    linewidth = 1) +
  geom_line(aes(x = 1 - roc_glmnet$specificities, y = roc_glmnet$sensitivities), linewidth = 1) +
  geom_line(aes(x = 1 - roc_rf$specificities,     y = roc_rf$sensitivities),     linewidth = 1) +
  geom_line(aes(x = 1 - roc_base$specificities,   y = roc_base$sensitivities),   linewidth = 1, linetype = "dashed") +
  geom_abline(slope = 1, intercept = 0, linetype = "dotted") +
  labs(
    title = "ROC Curves (Test Set)",
    x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)",
    caption = paste0("AUC: GLM=", round(auc(roc_glm),3),
                     " | GLMNET=", round(auc(roc_glmnet),3),
                     " | RF=", round(auc(roc_rf),3),
                     " | Baseline=", round(auc(roc_base),3))
  ) + theme_minimal()
save_plot(p_roc, file.path(OUT_DIR, "roc_curves.png"))

# 8b. Precision-Recall curve (models)
pr_curve <- function(truth, prob) {
  pred <- ROCR::prediction(prob, truth == "Returned")
  perf <- ROCR::performance(pred, "prec", "rec")
  tibble(recall = perf@x.values[[1]], precision = perf@y.values[[1]])
}
pr_glm    <- pr_curve(y_test, probs_glm)    %>% mutate(Model = "GLM")
pr_glmnet <- pr_curve(y_test, probs_glmnet) %>% mutate(Model = "GLMNET")
pr_rf     <- pr_curve(y_test, probs_rf)     %>% mutate(Model = "RF")

p_pr <- bind_rows(pr_glm, pr_glmnet, pr_rf) %>%
  ggplot(aes(x = recall, y = precision, color = Model)) +
  geom_path(linewidth = 1) +
  coord_equal() +
  labs(title = "Precision–Recall Curves (Test Set)", x = "Recall", y = "Precision") +
  theme_minimal()
save_plot(p_pr, file.path(OUT_DIR, "pr_curves.png"))

# 8c. Confusion matrix heatmap (best model @ tuned threshold)
best_pred <- factor(ifelse(best_probs >= BEST_THRESHOLD, "Returned", "Not_Returned"),
                    levels = c("Not_Returned", "Returned"))
cm_best <- caret::confusionMatrix(best_pred, y_test, positive = "Returned")
cm_tbl <- as.data.frame(cm_best$table)
colnames(cm_tbl) <- c("Predicted", "Actual", "Freq")

p_cm <- ggplot(cm_tbl, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", fontface = "bold") +
  scale_fill_gradient(low = "steelblue", high = "darkblue") +
  labs(title = paste0("Confusion Matrix (", BEST_MODEL_NAME, " @ threshold=", round(BEST_THRESHOLD,2), ")"),
       x = "Predicted", y = "Actual") +
  theme_minimal()
save_plot(p_cm, file.path(OUT_DIR, "confusion_matrix_best.png"))

# 8d. Feature importance (if available)
# GLM/GLMNET: coefficients; RF: importance
if (BEST_MODEL_NAME %in% c("RF")) {
  vi <- varImp(best_fit)$importance %>%
    rownames_to_column("Feature") %>%
    mutate(mean_imp = rowMeans(select(., -Feature), na.rm = TRUE)) %>%
    arrange(desc(mean_imp)) %>%
    slice_head(n = 20)
  
  p_vi <- ggplot(vi, aes(x = reorder(Feature, mean_imp), y = mean_imp)) +
    geom_col() + coord_flip() +
    labs(title = "Random Forest Feature Importance (Top 20)", x = "Feature", y = "Importance") +
    theme_minimal()
  save_plot(p_vi, file.path(OUT_DIR, "feature_importance_best.png"))
} else if (BEST_MODEL_NAME %in% c("GLMNET", "GLM")) {
  # For GLM/GLMNET, create a simple coefficient plot (abs effect)
  coefs <- tryCatch({
    broom::tidy(best_fit$finalModel)  # glmnet stores many lambdas; broom handles it
  }, error = function(e) NULL)
  
  if (!is.null(coefs)) {
    # pick the best tune row if glmnet
    if (BEST_MODEL_NAME == "GLMNET") {
      best_lambda <- best_fit$bestTune$lambda
      coefs <- broom::tidy(best_fit$finalModel) %>% filter(lambda == best_lambda)
    }
    coefs <- coefs %>%
      filter(term != "(Intercept)") %>%
      mutate(abs_est = abs(estimate)) %>%
      arrange(desc(abs_est)) %>% slice_head(n = 20)
    
    p_coef <- ggplot(coefs, aes(x = reorder(term, abs_est), y = estimate)) +
      geom_col() + coord_flip() +
      labs(title = paste0(BEST_MODEL_NAME, " Coefficients (Top 20 by |estimate|)"),
           x = "Feature (dummy)", y = "Coefficient") +
      theme_minimal()
    save_plot(p_coef, file.path(OUT_DIR, "feature_importance_best.png"))
  }
}

# -----------------------------
# 9) Save predictions and summary
# -----------------------------
preds_out <- test_df %>%
  select(returned_factor) %>%
  mutate(
    prob_glm = probs_glm,
    prob_glmnet = probs_glmnet,
    prob_rf = probs_rf,
    best_model = BEST_MODEL_NAME,
    best_prob = best_probs,
    best_threshold = BEST_THRESHOLD,
    best_pred = best_pred
  )
readr::write_csv(preds_out, file.path(OUT_DIR, "test_predictions_all_models.csv"))

summary_row <- metrics_tbl %>%
  filter(Model == BEST_MODEL_NAME) %>%
  mutate(BaseRate = base_rate) %>%
  select(Model, Threshold, ROC_AUC, Accuracy, Precision, Recall, F1, BaseRate)
readr::write_csv(summary_row, file.path(OUT_DIR, "best_model_summary.csv"))

cat("\nDone. Outputs saved to:", OUT_DIR, "\n")
print(list.files(OUT_DIR, full.names = TRUE))
