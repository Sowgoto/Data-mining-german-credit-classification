# German Credit Data Mining Classification Project

## Overview

This project applies data mining and machine learning techniques to the German Credit dataset. The main goal is to compare rule-based and tree-based classifiers using preprocessing, visualization, cross-validation, statistical testing, and robustness analysis.

The project includes two milestones. Milestone 1 focuses on data preprocessing, t-SNE visualization, classifier training, rule extraction, repeated 10-fold cross-validation, one-way ANOVA, and Tukey HSD post hoc testing. Milestone 2 evaluates classifier robustness by adding 10% random class-label noise and comparing model performance under noisy conditions.

## Dataset

The project uses the German Credit dataset.

Dataset file:

```text
data/credit-g.csv
```

## Project Tasks

## Milestone 1

### Task 1: Data Preprocessing and t-SNE Visualization

The dataset was preprocessed by:

- Converting character variables to factors
- Checking for missing values
- Normalizing numeric features
- Applying t-SNE for 2D visualization

The t-SNE visualization showed overlap between good and bad credit classes, indicating that the classification task is moderately challenging.

### Task 2: Classifier Models and Rule Bases

Three classifiers were trained and compared:

- Decision Tree using CART/rpart
- PART rule-based classifier
- RIPPER/JRip rule-based classifier

The rule bases were extracted and reviewed for interpretability.

### Task 3: Cross-Validation and Statistical Testing

The classifiers were evaluated using repeated 10-fold cross-validation with 3 repeats. One-way ANOVA and Tukey HSD post hoc tests were used to compare classifier accuracy.

## Milestone 2

### Task 4: Robustness to Noise

To test robustness, 10% random class-label noise was added to the dataset. The same repeated 10-fold cross-validation process was applied again.

The noisy-data comparison showed that:

- RIPPER achieved the highest mean accuracy under noise
- Decision Tree performed very closely to RIPPER
- PART performed significantly worse than Decision Tree and RIPPER

### Task 5: Final Classifier Selection

The classifiers were compared using four criteria:

- Rule perspicacity
- Mean accuracy
- Stability
- Robustness to noise

Based on the overall ranking, RIPPER was selected as the best general classifier for this dataset.

## Technologies Used

- R
- caret
- dplyr
- Rtsne
- ggplot2
- rpart
- partykit
- RWeka
- ANOVA
- Tukey HSD Test

## Repository Structure

```text
german-credit-data-mining-classification/
│
├── README.md
├── data/
│   └── credit-g.csv
│
├── code/
│   ├── Project_1_Milestone_1_code.R
│   └── Project_1_Milestone_2_code.R
│
└── reports/
    ├── Project_1_Milestone_1_Report.pdf
    └── Project_1_Milestone_2.pdf
```

## Requirements

Install the required R packages:

```r
install.packages("caret")
install.packages("dplyr")
install.packages("Rtsne")
install.packages("ggplot2")
install.packages("rpart")
install.packages("partykit")
install.packages("RWeka")
```

Note: `RWeka` may require Java to be installed and configured.

## How to Run

Open RStudio or an R terminal.

Set the working directory to the project folder.

Run Milestone 1:

```r
source("code/Project_1_Milestone_1_code.R")
```

Run Milestone 2:

```r
source("code/Project_1_Milestone_2_code.R")
```

Make sure the dataset is located at:

```text
data/credit-g.csv
```

If your R script expects `credit-g.csv` in the main folder, either move the dataset to the main folder or update the file path in the script.

## Results Summary

In Milestone 1, the classifiers were compared under clean data conditions. The ANOVA result showed no strong statistically significant difference among the classifier accuracies.

In Milestone 2, after adding 10% label noise, the classifiers showed different robustness levels. RIPPER achieved the best overall ranking based on accuracy, interpretability, stability, and robustness.

Final selected classifier:

```text
RIPPER / JRip
```

## Reports

The full project reports are included in the `reports/` folder.

## Disclaimer

This project is for academic and educational purposes only. It was completed as a data mining classification and statistical comparison project.

## Author

Sowgoto Raha Sunny  
M.S. in Cybersecurity  
University of North Texas
