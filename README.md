# San Francisco Crime Classification

This project predicts crime categories using historical crime data from San Francisco. It includes data preprocessing, exploratory data analysis (EDA), model training using CatBoost, and evaluation with visualizations and metrics.

## Project Structure

### Script Breakdown
1. **Data Loading:**
   - Load training and testing datasets from CSV files.
2. **Visualization Directory:**
   - Creates a directory (`plots`) to save all generated visualizations.
3. **Exploratory Data Analysis (EDA):**
   - Analyze trends in crime data, including:
     - Crimes per year, month, day of the week, and hour.
     - Heatmap of crimes over time (day vs. hour).
4. **Feature Engineering:**
   - Extracts time-based and address-related features from the dataset.
   - Encodes categorical features.
5. **Model Training:**
   - Trains a CatBoostClassifier using 5-fold stratified cross-validation.
   - Outputs metrics such as accuracy and F1 score for each fold.
6. **Evaluation:**
   - Generates a normalized confusion matrix for the top N crime categories.
   - Extracts and visualizes feature importance.
7. **Additional Visualizations:**
   - Crime category distribution.
   - Geographic crime distribution.
   - Trends over time.
   - Crime count by police district.
8. **Submission:**
   - Creates a submission CSV file for Kaggle.

## How to Run

### Prerequisites
- **Python 3.7+**
- Required libraries:
  - `pandas`
  - `numpy`
  - `catboost`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

Install dependencies:
```bash
pip install pandas numpy catboost scikit-learn matplotlib seaborn
```

### Steps to Execute
1. Place the `train.csv` and `test.csv` files in the `data` folder.
2. Run the script.
3. Generated visualizations will be saved in the `plots` directory.
4. Final predictions will be saved in `submission.csv`.

### File Paths
- Update the file paths for `train.csv` and `test.csv` in the script if necessary:
```python
train = pd.read_csv(r"C:\Users\sanka\sanfran\data\train.csv")
test = pd.read_csv(r"C:\Users\sanka\sanfran\data\test.csv")
```

## Outputs
### Visualizations
- **Yearly Crime Trends:** `plots/crimes_per_year.png`
- **Monthly Crime Trends:** `plots/crimes_per_month.png`
- **Crimes by Day of Week:** `plots/crimes_per_day_of_week.png`
- **Crimes by Hour:** `plots/crimes_per_hour.png`
- **Heatmap (Day vs Hour):** `plots/heatmap_crimes_day_hour.png`
- **Feature Importance:** `plots/feature_importance.png`
- **Top Crime Categories:** `plots/top_15_crime_categories.png`
- **Geographic Crime Distribution:** `plots/geographic_distribution.png`
- **Crimes Over Time:** `plots/crimes_over_time.png`
- **Crimes by District:** `plots/crimes_by_district.png`

### Metrics
- Stratified 5-fold cross-validation outputs:
  - Accuracy
  - F1 Score
  - Classification Report
  - Confusion Matrix (saved as `plots/confusion_matrix_top_N.png`)

### Submission
- `submission.csv`: Prediction probabilities for all classes, formatted for Kaggle submission.

## Key Features
- **Feature Engineering:** Extracts temporal and spatial features, encodes categorical variables, and incorporates domain knowledge (e.g., intersection detection).
- **Model:** CatBoostClassifier optimized for multi-class classification.
- **Evaluation:** Provides per-fold metrics and feature importance insights.
- **Visualizations:** Comprehensive crime analysis with multiple plots.

## Future Enhancements
- Optimize CatBoost hyperparameters further.
- Include additional external features (e.g., weather data).
- Experiment with other classification models and ensembles.

---

### Author
- **Sankalp**

Feel free to reach out for any queries or contributions!

