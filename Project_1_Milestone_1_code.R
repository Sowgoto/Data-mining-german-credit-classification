# Project 1 – Data Mining

# ======================= TASK 1 ===========================
# DATA PREPROCESSING + t-SNE

cat("\n================ TASK 1 =================\n")

library(caret)
library(dplyr)
library(Rtsne)
library(ggplot2)
set.seed(123) # for reproducibility

cat("Loading German Credit dataset...\n")
credit <- read.csv("credit-g.csv", stringsAsFactors = FALSE)

# Convert character variables to factors
credit[] <- lapply(credit, function(x) {
  if (is.character(x)) as.factor(x) else x
})

# Check missing values
cat("Missing values per column:\n")
print(colSums(is.na(credit)))

# Ensure class is factor
credit$class <- as.factor(credit$class)

# Normalize numeric features
numeric_cols <- sapply(credit, is.numeric)
preproc <- preProcess(credit[, numeric_cols], method = c("center", "scale"))

credit_preprocessed <- credit
credit_preprocessed[, numeric_cols] <- predict(preproc, credit[, numeric_cols])

# ----- t-SNE -----
cat("Running t-SNE...\n")

tsne_data <- credit_preprocessed
tsne_data[] <- lapply(tsne_data, function(x) {
  if (is.factor(x)) as.numeric(x) else x
})

tsne_result <- Rtsne(
  as.matrix(tsne_data[, -which(names(tsne_data) == "class")]),
  dims = 2,
  perplexity = 30,
  max_iter = 500,
  verbose = TRUE
)

tsne_df <- data.frame(
  Dim1 = tsne_result$Y[,1],
  Dim2 = tsne_result$Y[,2],
  Class = credit_preprocessed$class
)

ggplot(tsne_df, aes(Dim1, Dim2, color = Class)) +
  geom_point(alpha = 0.7) +
  theme_minimal() +
  ggtitle("t-SNE Visualization of German Credit Dataset")

# ======================= TASK 2 ===========================
# CLASSIFIER MODELS + RULE BASES

cat("\n================ TASK 2 =================\n")

library(rpart)
library(partykit)
library(RWeka)

# Train/Test split
set.seed(123)
train_index <- createDataPartition(
  credit_preprocessed$class,
  p = 0.7,
  list = FALSE
)

train_data <- credit_preprocessed[train_index, ]
test_data  <- credit_preprocessed[-train_index, ]

# ----- Train models -----
cat("Training Decision Tree...\n")
dt_model <- rpart(class ~ ., data = train_data, method = "class")

cat("Training PART...\n")
part_model <- PART(class ~ ., data = train_data)

cat("Training RIPPER...\n")
ripper_model <- JRip(class ~ ., data = train_data)

# ----- Rule bases -----
cat("\nDecision Tree Rules:\n")
print(dt_model)

cat("\nPART Rules:\n")
print(part_model)

cat("\nRIPPER Rules:\n")
print(ripper_model)

# ======================= TASK 3 ===========================
# 10-FOLD CV + ANOVA

cat("\n================ TASK 3 =================\n")

library(partykit)

ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  savePredictions = "final"
)

# Decision Tree CV
cat("Cross-validating Decision Tree...\n")
dt_cv <- train(
  class ~ .,
  data = credit_preprocessed,
  method = "rpart",
  trControl = ctrl
)

# RIPPER CV
cat("Cross-validating RIPPER...\n")
ripper_cv <- train(
  class ~ .,
  data = credit_preprocessed,
  method = "JRip",
  trControl = ctrl
)

# Manual 10-fold CV repeated 3 times for PART (ctree)
set.seed(123)
repeats <- 3
folds <- 10
part_acc <- numeric(repeats * folds)

for (r in 1:repeats) {
  fold_indices <- createFolds(credit_preprocessed$class, k = folds, list = TRUE)
  
  for (f in 1:folds) {
    test_indices <- fold_indices[[f]]
    train_indices <- setdiff(seq_len(nrow(credit_preprocessed)), test_indices)
    
    train_fold <- credit_preprocessed[train_indices, ]
    test_fold <- credit_preprocessed[test_indices, ]
    
    model <- PART(class ~ ., data = train_fold)
    preds <- predict(model, newdata = test_fold)
    
    part_acc[(r - 1) * folds + f] <- mean(preds == test_fold$class)
  }
}

cat("PART cross-validation accuracies (30 folds):\n")
print(part_acc)

# Collect caret accuracies
dt_acc     <- dt_cv$resample$Accuracy
ripper_acc <- ripper_cv$resample$Accuracy

num_resamples <- length(dt_acc)  # should be 30

# Combine accuracy values
accuracy <- c(dt_acc, part_acc, ripper_acc)
classifier <- factor(rep(
  c("DecisionTree", "PART", "RIPPER"),
  each = num_resamples
))
anova_data <- data.frame(accuracy, classifier)

cat("\nPerforming One-Way ANOVA...\n")
anova_result <- aov(accuracy ~ classifier, data = anova_data)
print(summary(anova_result))

cat("\nPerforming Tukey HSD Test...\n")
tukey_result <- TukeyHSD(anova_result)
print(tukey_result)
