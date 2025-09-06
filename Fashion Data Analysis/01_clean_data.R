# 01_clean_data.R
# Purpose: clean the raw retail fashion dataset and produce one cleaned dataset
# Outputs: cleaned_fashion_boutique.csv, cleaned_fashion_boutique.rds, data_dictionary.csv, cleaning_log.txt
# Run in RStudio. Edit RAW_DATA_PATH below if your CSV is in a different location.

# -------------------------
# 0. Setup
# -------------------------


packages <- c("tidyverse", "lubridate", "janitor", "skimr", "data.table")
for(p in packages) if(!requireNamespace(p, quietly = TRUE)) install.packages(p)
library(tidyverse); library(lubridate); library(janitor); library(skimr); library(data.table)


# File paths (edit if needed)

RAW_DATA_PATH <- "C:/Users/Simphiwe Mathosa/OneDrive - belgiumcampus.ac.za/Documents/Fashion Data Analysis/fashion_boutique_dataset.csv"   # default path 
if(!file.exists(RAW_DATA_PATH)) {
  message("Default raw file not found. Please choose the file interactively.")
  RAW_DATA_PATH <- file.choose()  # interactive fallback
}
CLEANED_CSV <- "cleaned_fashion_boutique.csv"
CLEANED_RDS <- "cleaned_fashion_boutique.rds"
DATA_DICT_CSV <- "data_dictionary.csv"
LOG_FILE <- "cleaning_log.txt"
REPORT_FILE <- "data_quality_report.html"  # optional


# Simple logger
log_lines <- c()
log <- function(...) {
  msg <- paste0("[", Sys.time(), "] ", paste0(..., collapse=""))
  message(msg)
  log_lines <<- c(log_lines, msg)
}

log("Starting cleaning script.")
raw <- readr::read_csv(RAW_DATA_PATH, guess_max = 2000)
log("Raw rows:", nrow(raw), "cols:", ncol(raw))


# -------------------------
# 1. Initial exploration
# -------------------------
raw <- raw %>% clean_names()  # lower_case column names
log("Columns after clean_names:", paste(colnames(raw), collapse=", "))

# quick skim
skim_out <- skimr::skim(raw)


# -------------------------
# 2. Remove exact duplicate rows
# -------------------------
n_before <- nrow(raw)
raw <- raw %>% distinct()
n_after <- nrow(raw)
log("Removed exact duplicate rows:", n_before - n_after)


# -------------------------
# 3. Standardize text columns
# -------------------------
raw <- raw %>%
  mutate(
    brand = if_else(!is.na(brand), str_squish(str_to_title(brand)), brand),
    category = if_else(!is.na(category), str_squish(str_to_title(category)), category),
    color = if_else(!is.na(color), str_squish(str_to_title(color)), color),
    return_reason = if_else(!is.na(return_reason), str_squish(str_to_title(return_reason)), return_reason)
  )


# -------------------------
# 4. Parse dates
# -------------------------
raw <- raw %>%
  mutate(purchase_date = parse_date_time(purchase_date, orders = c("ymd","ymd HMS","dmy","mdy","Y-m-d")) %>% as.Date())

bad_dates <- sum(is.na(raw$purchase_date))
if(bad_dates > 0) log("Date parsing produced NAs:", bad_dates, " — investigate these rows.")
# Optionally filter out rows with NA date, but here we keep and log.



# -------------------------
# 5. Correct types & basic validation
# -------------------------
clean <- raw %>%
  mutate(
    product_id = as.character(product_id),
    category = as.factor(category),
    brand = as.factor(brand),
    season = as.factor(season),
    size = as.character(size),
    color = as.factor(color),
    original_price = as.numeric(original_price),
    markdown_percentage = as.numeric(markdown_percentage),
    current_price = as.numeric(current_price),
    stock_quantity = as.integer(stock_quantity),
    customer_rating = as.numeric(customer_rating),
    is_returned = case_when(
      tolower(as.character(is_returned)) %in% c("true","t","1","yes") ~ TRUE,
      tolower(as.character(is_returned)) %in% c("false","f","0","no") ~ FALSE,
      TRUE ~ NA
    ),
    return_reason = as.character(return_reason)
  )


# -------------------------
# 6. Data quality rules & fixes
# -------------------------
# Clamp markdown_percentage to [0,100]
clean <- clean %>% mutate(markdown_percentage = ifelse(is.na(markdown_percentage), 0, markdown_percentage))
clean <- clean %>% mutate(
  markdown_percentage = pmin(pmax(markdown_percentage, 0), 100)
)



# Compute expected current price and fix inconsistencies
clean <- clean %>%
  mutate(
    expected_current_price = round(original_price * (1 - markdown_percentage/100), 2),
    price_discrepancy = if_else(abs(current_price - expected_current_price) > 0.01, TRUE, FALSE),
    current_price = if_else(price_discrepancy, expected_current_price, current_price)
  )

log("Price discrepancies flagged:", sum(clean$price_discrepancy, na.rm = TRUE))



# Handle sizes: accessories have NA -> "Not Applicable"; other NAs -> "Unknown"
clean <- clean %>%
  mutate(
    size = case_when(
      is.na(size) & category == "Accessories" ~ "Not Applicable",
      is.na(size) & category != "Accessories" ~ "Unknown",
      TRUE ~ size
    ),
    size = as.factor(size)
  )


# Handle return_reason for non-returned rows & returned missing reasons
clean <- clean %>%
  mutate(
    return_reason = case_when(
      is.na(is_returned) ~ return_reason, # leave as-is if return flag unknown
      is_returned == FALSE ~ "Not Returned",
      is_returned == TRUE & is.na(return_reason) ~ "Unknown",
      TRUE ~ return_reason
    ),
    return_reason = as.factor(return_reason)
  )




# Impute customer_rating with median of category; create rating_imputed flag
median_by_cat <- clean %>%
  group_by(category) %>%
  summarise(median_rating = median(customer_rating, na.rm = TRUE)) %>%
  mutate(median_rating = ifelse(is.na(median_rating), NA_real_, median_rating))

overall_median_rating <- median(clean$customer_rating, na.rm = TRUE)



# join and impute
clean <- clean %>%
  left_join(median_by_cat, by = "category") %>%
  mutate(
    customer_rating = ifelse(is.na(customer_rating), coalesce(median_rating, overall_median_rating), customer_rating),
    rating_imputed = ifelse(is.na(median_rating) & is.na(customer_rating), TRUE, FALSE)
  ) %>%
  select(-median_rating)
clean$rating_imputed[is.na(clean$rating_imputed)] <- FALSE





# Feature engineering
clean <- clean %>%
  mutate(
    month = floor_date(purchase_date, "month"),
    month_year = format(purchase_date, "%Y-%m"),
    year = year(purchase_date),
    quarter = paste0(year(purchase_date), " Q", quarter(purchase_date)),
    week = floor_date(purchase_date, "week"),
    discounted = markdown_percentage > 0,
    discount_amount = round(original_price - current_price, 2),
    profit_margin = round(current_price - original_price, 2)
  )






# -------------------------
# 7. Final validation assertions
# -------------------------
# Ensure no NA in critical columns we decided to fill:
critical_cols <- c("product_id","category","brand","original_price","current_price","purchase_date")
nas_critical <- sapply(clean[critical_cols], function(x) sum(is.na(x)))
log("NAs in critical cols:", paste(names(nas_critical)[nas_critical>0], nas_critical[nas_critical>0], collapse = "; "))




# Sanity checks
if(any(clean$original_price <= 0, na.rm = TRUE)) log("Warning: non-positive original_price values found.")
if(any(clean$current_price < 0, na.rm = TRUE)) log("Warning: negative current_price found.")



# -------------------------
# 8. Save outputs & documentation
# -------------------------
data.table::fwrite(clean, CLEANED_CSV)
saveRDS(clean, CLEANED_RDS)
log("Saved cleaned dataset to:", CLEANED_CSV, "and", CLEANED_RDS)



# Write data dictionary (column, type, short description)
col_desc <- tibble(
  column = colnames(clean),
  type = sapply(clean, class),
  description = ""  # you can fill descriptions manually or programmatically
)
data.table::fwrite(col_desc, DATA_DICT_CSV)
log("Wrote data dictionary template to", DATA_DICT_CSV)




#write cleaning_log
writeLines(log_lines, LOG_FILE)
log("Wrote cleaning log to", LOG_FILE)



# optional: produce a skimr summary as HTML-ish (write to csv)
skim_summary <- skimr::skim(clean)
# save a small summary as .RDS or CSV for quick inspection
saveRDS(skim_summary, file = "skim_summary.rds")



# save session info
capture.output(sessionInfo(), file = "session_info.txt")



log("Cleaning complete. Rows in cleaned data:", nrow(clean))

# End script

























