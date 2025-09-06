# 03_forecast_seasonal_sales.R
# Objective: Forecast seasonal sales using the cleaned dataset
# Inputs: cleaned_fashion_boutique.csv (from 01_clean_data.R)
# Outputs:
#   - outputs_forecast/
#       monthly_revenue.csv
#       monthly_revenue_wide.csv (year-month matrix for heatmap)
#       decomposition_overall.png
#       monthly_revenue_line.png
#       seasonal_subseries.png
#       calendar_heatmap.png
#       forecast_overall_{MODEL}.png
#       forecast_overall_best.png
#       forecast_overall_best.csv
#       <per-category/brand> analogous files
# Notes: Uses columns from the data dictionary:
#   purchase_date (Date), month (Date), current_price (numeric),
#   category (factor), brand (factor), season (factor), year (int), month_year (char)




# -----------------------------
# 0) Setup
# -----------------------------
packages <- c("tidyverse", "lubridate", "forecast", "zoo", "scales", "patchwork")
for (p in packages) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
library(tidyverse)
library(lubridate)
library(forecast)
library(zoo)
library(scales)
library(patchwork)



CLEANED_DATA_PATH <- "cleaned_fashion_boutique.csv"
OUT_DIR <- "outputs_forecast"
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)


# Helper: Save ggplot safely
save_plot <- function(plot, filename, width = 9, height = 6) {
  ggsave(filename, plot, width = width, height = height, dpi = 300)
}



df <- readr::read_csv("cleaned_fashion_boutique.csv")

cat("=== DATA EXPLORATION ===\n")
cat("Rows:", nrow(df), " | Columns:", ncol(df), "\n\n")
cat("Column names:\n"); print(colnames(df))
cat("\nSample:\n"); print(head(df, 5))
cat("\nMissing values per column:\n"); print(colSums(is.na(df)))




# Sanity checks
stopifnot("purchase_date must be Date" = inherits(df$purchase_date, "Date"))
stopifnot("current_price must be numeric" = is.numeric(df$current_price))
if (!("month" %in% names(df)) || !inherits(df$month, "Date")) {
  # fallback compute month if not present for any reason
  df <- df %>% mutate(month = floor_date(purchase_date, "month"))
}





# -----------------------------
# 2) Build monthly revenue series (overall)
#    - ensure a complete monthly sequence (fill gaps with 0)
# -----------------------------
monthly <- df %>%
  group_by(month) %>%
  summarise(revenue = sum(current_price, na.rm = TRUE), .groups = "drop") %>%
  arrange(month)

# complete sequence
all_months <- tibble(month = seq.Date(from = min(monthly$month, na.rm = TRUE),
                                      to   = max(monthly$month, na.rm = TRUE),
                                      by   = "month"))
monthly <- all_months %>%
  left_join(monthly, by = "month") %>%
  mutate(revenue = coalesce(revenue, 0))






# Exports for Power BI and auditing
readr::write_csv(monthly, file.path(OUT_DIR, "monthly_revenue.csv"))


# Wide matrix (Year x Month number) for heatmap
monthly_wide <- monthly %>%
  mutate(Y = year(month), M = month(month)) %>%
  select(Y, M, revenue) %>%
  pivot_wider(names_from = M, values_from = revenue, values_fill = 0) %>%
  arrange(Y)
readr::write_csv(monthly_wide, file.path(OUT_DIR, "monthly_revenue_wide.csv"))









# -----------------------------
# 3) Visualize raw seasonal patterns
# -----------------------------
p_line <- ggplot(monthly, aes(x = month, y = revenue)) +
  geom_line() +
  geom_smooth(method = "loess", se = FALSE, span = 0.3) +
  scale_y_continuous(labels = dollar_format()) +
  labs(title = "Monthly Revenue (Overall)", x = "Month", y = "Revenue") +
  theme_minimal()
save_plot(p_line, file.path(OUT_DIR, "monthly_revenue_line.png"))

# Seasonal subseries (month-of-year patterns)
monthly <- monthly %>%
  mutate(Year = year(month), MonthName = factor(month(month, label = TRUE, abbr = TRUE),
                                                levels = month.abb))

p_subseries <- ggplot(monthly, aes(x = Year, y = revenue, group = MonthName)) +
  geom_line(alpha = 0.7) +
  facet_wrap(~ MonthName, ncol = 4, scales = "free_y") +
  scale_y_continuous(labels = dollar_format()) +
  labs(title = "Seasonal Subseries (Month-of-Year Patterns)", x = "Year", y = "Revenue") +
  theme_minimal()
save_plot(p_subseries, file.path(OUT_DIR, "seasonal_subseries.png"))

# Calendar-style heatmap (Year vs Month)
p_heat <- monthly %>%
  mutate(MonthNum = month(month)) %>%
  ggplot(aes(x = MonthNum, y = factor(Year), fill = revenue)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue", labels = dollar_format()) +
  scale_x_continuous(breaks = 1:12, labels = month.abb) +
  labs(title = "Revenue Heatmap by Year-Month", x = "Month", y = "Year", fill = "Revenue") +
  theme_minimal()
save_plot(p_heat, file.path(OUT_DIR, "calendar_heatmap.png"))

# -----------------------------
# 4) STL decomposition (overall)
# -----------------------------
# Build ts object (monthly frequency = 12)
start_ym <- as.yearmon(min(monthly$month))
ts_start <- c(as.integer(format(as.Date(start_ym), "%Y")),
              as.integer(format(as.Date(start_ym), "%m")))
y <- ts(monthly$revenue, frequency = 12, start = ts_start)

fit_stl <- stl(y, s.window = "periodic")
png(file.path(OUT_DIR, "decomposition_overall.png"), width = 1200, height = 800, res = 150)
plot(fit_stl, main = "STL Decomposition - Overall Monthly Revenue")
dev.off()

# -----------------------------
# 5) Train/test split (last 6 months as holdout)
# -----------------------------
h <- min(12, max(6, ceiling(length(y) * 0.15)))  # forecast horizon (try 12; at least 6)
if (length(y) < 24) h <- min(6, length(y) %/% 4) # safety for very short series
train <- window(y, end = time(y)[length(y) - h])
test  <- window(y, start = time(y)[length(y) - h + 1])

# Models
fit_ets   <- ets(train)
fit_arima <- auto.arima(train, stepwise = FALSE, approximation = FALSE)
fit_snaive <- snaive(train)

fc_ets    <- forecast(fit_ets, h = length(test))
fc_arima  <- forecast(fit_arima, h = length(test))
fc_snaive <- forecast(fit_snaive, h = length(test))

acc_ets    <- accuracy(fc_ets, test)
acc_arima  <- accuracy(fc_arima, test)
acc_snaive <- accuracy(fc_snaive, test)

acc_tbl <- bind_rows(
  tibble(Model = "ETS",    RMSE = acc_ets["Test set","RMSE"],   MAPE = acc_ets["Test set","MAPE"]),
  tibble(Model = "ARIMA",  RMSE = acc_arima["Test set","RMSE"], MAPE = acc_arima["Test set","MAPE"]),
  tibble(Model = "SNAIVE", RMSE = acc_snaive["Test set","RMSE"],MAPE = acc_snaive["Test set","MAPE"])
) %>% arrange(MAPE)

print(acc_tbl)
best_model_name <- acc_tbl$Model[1]
cat("\nBest model by MAPE:", best_model_name, "\n")

# -----------------------------
# 6) Refit best model on full series & forecast 12 months
# -----------------------------
fit_full <- switch(best_model_name,
                   "ETS"    = ets(y),
                   "ARIMA"  = auto.arima(y, stepwise = FALSE, approximation = FALSE),
                   "SNAIVE" = snaive(y))
fc_full <- forecast(fit_full, h = 12)

# Plot all model forecasts (for comparison)
p_fc_ets <- autoplot(forecast(ets(y), h = 12)) +
  labs(title = "Forecast - ETS (Overall)", x = "", y = "Revenue") +
  scale_y_continuous(labels = dollar_format()) + theme_minimal()
save_plot(p_fc_ets, file.path(OUT_DIR, "forecast_overall_ETS.png"))

p_fc_arima <- autoplot(forecast(auto.arima(y, stepwise = FALSE, approximation = FALSE), h = 12)) +
  labs(title = "Forecast - ARIMA (Overall)", x = "", y = "Revenue") +
  scale_y_continuous(labels = dollar_format()) + theme_minimal()
save_plot(p_fc_arima, file.path(OUT_DIR, "forecast_overall_ARIMA.png"))

p_fc_snaive <- autoplot(forecast(snaive(y), h = 12)) +
  labs(title = "Forecast - Seasonal Naive (Overall)", x = "", y = "Revenue") +
  scale_y_continuous(labels = dollar_format()) + theme_minimal()
save_plot(p_fc_snaive, file.path(OUT_DIR, "forecast_overall_SNAIVE.png"))

# Plot best-only forecast
p_fc_best <- autoplot(fc_full) +
  labs(title = paste0("Forecast - ", best_model_name, " (Overall)"),
       x = "", y = "Revenue") +
  scale_y_continuous(labels = dollar_format()) +
  theme_minimal()
save_plot(p_fc_best, file.path(OUT_DIR, "forecast_overall_best.png"))

# Save best forecast values (point forecast + intervals)
best_df <- tibble(
  date = seq.Date(from = max(monthly$month) %m+% months(1),
                  by = "month", length.out = length(fc_full$mean)),
  point_forecast = as.numeric(fc_full$mean),
  lo80 = as.numeric(fc_full$lower[,"80%"]),
  hi80 = as.numeric(fc_full$upper[,"80%"]),
  lo95 = as.numeric(fc_full$lower[,"95%"]),
  hi95 = as.numeric(fc_full$upper[,"95%"]),
  model = best_model_name
)
readr::write_csv(best_df, file.path(OUT_DIR, "forecast_overall_best.csv"))

# -----------------------------
# 7) Seasonal context views by 'season' field (Spring/Summer/Fall/Winter)
# -----------------------------
season_monthly <- df %>%
  group_by(month, season) %>%
  summarise(revenue = sum(current_price, na.rm = TRUE), .groups = "drop")

p_season_trend <- ggplot(season_monthly, aes(x = month, y = revenue, color = season)) +
  geom_line() +
  scale_y_continuous(labels = dollar_format()) +
  labs(title = "Monthly Revenue by Season", x = "Month", y = "Revenue") +
  theme_minimal()
save_plot(p_season_trend, file.path(OUT_DIR, "monthly_revenue_by_season.png"))

# -----------------------------
# 8) Forecast for Top 3 Categories and Top 3 Brands
# -----------------------------
make_series_and_forecast <- function(data, key_col, key_value, out_prefix) {
  series <- data %>%
    filter(.data[[key_col]] == key_value) %>%
    group_by(month) %>%
    summarise(revenue = sum(current_price, na.rm = TRUE), .groups = "drop") %>%
    arrange(month)
  
  if (nrow(series) < 6) return(invisible(NULL))  # too short to forecast
  
  # complete months (fill 0)
  all_m <- tibble(month = seq.Date(min(series$month), max(series$month), by = "month"))
  series <- all_m %>% left_join(series, by = "month") %>% mutate(revenue = coalesce(revenue, 0))
  
  # ts
  start_ym <- as.yearmon(min(series$month))
  ts_start <- c(as.integer(format(as.Date(start_ym), "%Y")),
                as.integer(format(as.Date(start_ym), "%m")))
  yy <- ts(series$revenue, frequency = 12, start = ts_start)
  
  # choose model by AICc on full series as a quick heuristic (shorter series)
  fit_e  <- ets(yy)
  fit_a  <- auto.arima(yy, stepwise = FALSE, approximation = FALSE)
  fit_sn <- snaive(yy)
  
  cand <- list(ETS = fit_e, ARIMA = fit_a, SNAIVE = fit_sn)
  aicc <- sapply(cand, AICc)
  best <- names(which.min(aicc))
  fc   <- forecast(cand[[best]], h = 12)
  
  # plots
  p_line <- ggplot(series, aes(x = month, y = revenue)) +
    geom_line() +
    geom_smooth(method = "loess", se = FALSE, span = 0.3) +
    scale_y_continuous(labels = dollar_format()) +
    labs(title = paste0("Monthly Revenue - ", key_col, ": ", key_value),
         x = "Month", y = "Revenue") + theme_minimal()
  
  p_fc   <- autoplot(fc) +
    labs(title = paste0("Forecast (", best, ") - ", key_col, ": ", key_value),
         x = "", y = "Revenue") +
    scale_y_continuous(labels = dollar_format()) + theme_minimal()
  
  save_plot(p_line, file.path(OUT_DIR, paste0(out_prefix, "_line.png")))
  save_plot(p_fc,   file.path(OUT_DIR, paste0(out_prefix, "_forecast.png")))
  
  # export forecasts
  out_df <- tibble(
    date = seq.Date(from = max(series$month) %m+% months(1),
                    by = "month", length.out = length(fc$mean)),
    point_forecast = as.numeric(fc$mean),
    lo80 = as.numeric(fc$lower[,"80%"]),
    hi80 = as.numeric(fc$upper[,"80%"]),
    lo95 = as.numeric(fc$lower[,"95%"]),
    hi95 = as.numeric(fc$upper[,"95%"]),
    model = best,
    key = key_value,
    key_type = key_col
  )
  readr::write_csv(out_df, file.path(OUT_DIR, paste0(out_prefix, "_forecast.csv")))
}

# Top 3 categories by total revenue
top3_cat <- df %>%
  group_by(category) %>%
  summarise(revenue = sum(current_price, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(revenue)) %>%
  slice_head(n = 3) %>%
  pull(category) %>% as.character()

# Top 3 brands by total revenue
top3_brand <- df %>%
  group_by(brand) %>%
  summarise(revenue = sum(current_price, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(revenue)) %>%
  slice_head(n = 3) %>%
  pull(brand) %>% as.character()

for (c in top3_cat) {
  make_series_and_forecast(df, "category", c, out_prefix = paste0("category_", make.names(c)))
}
for (b in top3_brand) {
  make_series_and_forecast(df, "brand", b, out_prefix = paste0("brand_", make.names(b)))
}

cat("\nForecasting complete. Outputs saved to folder:", OUT_DIR, "\n")

# End of script






















