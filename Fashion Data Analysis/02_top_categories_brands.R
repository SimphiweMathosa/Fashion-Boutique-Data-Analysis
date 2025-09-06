# 02_top_categories_brands.R
# Purpose: Identify top-performing categories and brands in the Fashion Boutique dataset
# Inputs: cleaned_fashion_boutique.csv (output from 01_clean_data.R)
# Outputs: Summary tables, ggplot visualizations, and optional export CSVs
# Author: [Your Name]
# Date: [Date]


# -------------------------
# 0. Setup
# -------------------------
packages <- c("tidyverse", "skimr")
for(p in packages) if(!requireNamespace(p, quietly = TRUE)) install.packages(p)
library(tidyverse)
library(skimr)


# File paths
CLEANED_DATA_PATH <- "cleaned_fashion_boutique.csv"
OUTPUT_FOLDER <- "outputs_top_categories_brands"
if(!dir.exists(OUTPUT_FOLDER)) dir.create(OUTPUT_FOLDER)





# -------------------------
# 1. Load Data
# -------------------------
df <- read_csv(CLEANED_DATA_PATH)





# -------------------------
# 2. Basic Data Exploration
# -------------------------
cat("Dataset dimensions: ", nrow(df), " rows and ", ncol(df), " columns\n\n")
cat("Column names:\n")
print(colnames(df))



cat("\nData types summary:\n")
print(str(df))



cat("\nQuick skim summary:\n")
skim(df)



# Missing values summary
cat("\nMissing values per column:\n")
print(colSums(is.na(df)))





# -------------------------
# 3. Exploratory Analysis - Categories & Brands
# -------------------------

# Revenue by Category
category_summary <- df %>%
  group_by(category) %>%
  summarise(
    total_revenue = sum(current_price, na.rm = TRUE),
    avg_price = mean(current_price, na.rm = TRUE),
    total_units_sold = n(),
    return_rate = mean(is_returned, na.rm = TRUE)
  ) %>%
  arrange(desc(total_revenue))

print(category_summary)




# Revenue by Brand
brand_summary <- df %>%
  group_by(brand) %>%
  summarise(
    total_revenue = sum(current_price, na.rm = TRUE),
    avg_price = mean(current_price, na.rm = TRUE),
    total_units_sold = n(),
    return_rate = mean(is_returned, na.rm = TRUE)
  ) %>%
  arrange(desc(total_revenue))

print(brand_summary)




# -------------------------
# 4. Visualizations
# -------------------------

library(ggplot2)

# Top Categories by Revenue
p1 <- category_summary %>%
  ggplot(aes(x = reorder(category, total_revenue), y = total_revenue, fill = category)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  labs(title = "Top Categories by Total Revenue", x = "Category", y = "Revenue ($)") +
  theme_minimal()

ggsave(file.path(OUTPUT_FOLDER, "top_categories_revenue.png"), p1, width=7, height=5)



# Top Brands by Revenue
p2 <- brand_summary %>%
  ggplot(aes(x = reorder(brand, total_revenue), y = total_revenue, fill = brand)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  labs(title = "Top Brands by Total Revenue", x = "Brand", y = "Revenue ($)") +
  theme_minimal()

ggsave(file.path(OUTPUT_FOLDER, "top_brands_revenue.png"), p2, width=7, height=5)




# Return Rate by Category
p3 <- category_summary %>%
  ggplot(aes(x = reorder(category, return_rate), y = return_rate, fill = category)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Return Rate by Category", x = "Category", y = "Return Rate") +
  theme_minimal()

ggsave(file.path(OUTPUT_FOLDER, "return_rate_category.png"), p3, width=7, height=5)





# Return Rate by Brand
p4 <- brand_summary %>%
  ggplot(aes(x = reorder(brand, return_rate), y = return_rate, fill = brand)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Return Rate by Brand", x = "Brand", y = "Return Rate") +
  theme_minimal()

ggsave(file.path(OUTPUT_FOLDER, "return_rate_brand.png"), p4, width=7, height=5)





# -------------------------
# 5. Save Summaries
# -------------------------
write_csv(category_summary, file.path(OUTPUT_FOLDER, "category_summary.csv"))
write_csv(brand_summary, file.path(OUTPUT_FOLDER, "brand_summary.csv"))

cat("\nAnalysis complete. Results saved in:", OUTPUT_FOLDER, "\n")
