### Metrics of complexity instead of MGN's one
# phonotactic surpisal?
# LDL NED on unseen lemmas
# Try Qumin again? 

# For LSTM need to sub-sample the data to make data sizes comparable (think also about the same diversity of the data)

library(dplyr)
library(rstan)
library(cmdstanr)
library(brms)
library(lingtypology)

mgn.data <- read.csv("mgn_data/results-final/all-accuracies.csv")
pop.data <- read.csv("ethnologue_population_data.csv")


unique(mgn.data$lang)
# ---------------------------------------------------------
# 1. Merge and Filter Datasets
# ---------------------------------------------------------

# Filter morphological data for 200 lemmas
mgn_sub <- mgn.data %>% 
  filter(num == 200)

# Merge datasets using the ISO code
# 'lang' in mgn.data corresponds to 'ISO' in pop.data
data_merged <- inner_join(mgn_sub, pop.data, by = c("lang" = "ISO"))

# Drop rows with missing values in our variables of interest
data_merged <- data_merged %>%
  filter(!is.na(L2prop), !is.na(Population)) %>%
  mutate(language_loc_name = case_match(Language,
    "Catalan-Valencian-Balear" ~ "Catalan",
    "Crimean Turkish"          ~ "Crimean Gothic",
    "Irish Gaelic"             ~ "Irish",
    "Standard German"          ~ "German",
    .default = Language # Will be needed for spatial info
  ))


# ---------------------------------------------------------
# 2. Add Spatial Information
# ---------------------------------------------------------

# Manually replace some entries to use `lat.lang` and `long.lang`:


# Fetch latitude and longitude using lingtypology
data_merged$lat <- lat.lang(data_merged$language_loc_name)
data_merged$lon <- long.lang(data_merged$language_loc_name)

# Remove any languages where Glottolog coordinates are unavailable
data_merged <- data_merged %>% 
  filter(!is.na(lat) & !is.na(lon))


# ---------------------------------------------------------
# 3. Variable Transformation
# ---------------------------------------------------------

# Log-transform population size as indicated by Koplenig
# This is required to control for the confounding effect of population scale
data_merged$log_pop <- log(data_merged$Population)

# ---------------------------------------------------------
# 4. Bayesian Model Setup with brms
# ---------------------------------------------------------

# We use a Zero-One-Inflated Beta (zoib) family because the 
# analogical complexity score (mean) contains exact 1s.
# 
# gp(lat, lon) fits a Gaussian Process over the coordinates 
# to control for spatial/areal autocorrelation.

formula <- bf(
  mean ~ L2prop + log_pop + (1 | Family) + (1 | pos) + (1 | lang) + gp(lat, lon)
)

# Define basic vague/weakly informative priors
priors <- c(
  prior(normal(0, 5), class = "b"),           # Population-level effects (slopes)
  prior(normal(0, 10), class = "Intercept"),  # Global intercept
  prior(exponential(1), class = "sd"),  # Standard deviations for random effects
  prior(student_t(3, 0, 2.5), class = "sdgp"),# Standard deviation for the Gaussian Process
  prior(gamma(0.01, 0.01), class = "phi")     # Precision parameter for the Beta distribution
)

# Fit the model
# (Adjust cores and iterations based on your hardware capabilities)
model_brm <- brm(
  formula = formula,
  data = data_merged,
  family = zero_one_inflated_beta(),
  prior = priors,
  chains = 4,
  cores = 4,
  threads = threading(2),
  backend = "cmdstanr", 
  iter = 4000,
  warmup = 2250,
  control = list(adapt_delta = 0.99) # Helps with complex GP and ZOIB mixing
)

# Check model summary
summary(model_brm)
