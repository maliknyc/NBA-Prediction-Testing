# Import necessary libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

# Function to run the model with a dynamic target
def run_model_with_target(target_points):
    # Load dataset
    jb = pd.read_csv("https://raw.githubusercontent.com/maliknyc/NBA-Prediction-Testing/main/JaylenBrownTest.csv")

    # Preprocess the dataset
    jb['Date'] = pd.to_datetime(jb['Date'], format='%m/%d/%Y')
    jb = jb.rename(columns={'3PA': 'TPA', '3P%': 'TPP'})
    jb = jb.sort_values('Date')
    jb_cleaned = jb[['Date', 'Opp', 'MP', 'FGA', 'TPA', 'TPP', 'TRB', 'AST', 'STL', 'PTS', 'PRA', 'PR', 'PA', 'RA', 'SB']]

    # Add average points vs team
    avg_points_vs_team = jb.groupby('Opp')['PTS'].mean().reset_index().rename(columns={'PTS': 'avgPTS_vteam'})
    jb_cleaned = jb_cleaned.merge(avg_points_vs_team, on='Opp', how='left')

    # Add defensive ratings
    def_ratings = pd.DataFrame({
        'Opp': ["BOS", "DEN", "OKC", "MIN", "LAC", "DAL", "NYK", "MIL", "NOP", "PHO", "CLE", "IND", "LAL", "ORL", "PHI", "GSW", "MIA", "SAC", "HOU", "CHI", "ATL", "BRK", "UTA", "MEM", "TOR", "SAS", "CHO", "POR", "WAS", "DET"],
        'DEF_RTG': [110.6, 112.3, 111.0, 110.4, 113.1, 114.9, 111.4, 110.2, 111.9, 113.7, 115.0, 117.6, 114.8, 110.8, 113.0, 114.5, 111.5, 114.4, 112.8, 115.7, 115.4, 114.6, 119.6, 113.7, 115.6, 116.1, 116.9, 118.0, 118.6, 118.0]
    })
    jb_cleaned = jb_cleaned.merge(def_ratings, on='Opp', how='left')

    # Calculate days of rest and cap at 30 days
    jb_cleaned['days_rest'] = jb_cleaned['Date'].diff().dt.days.fillna(0).astype(int)
    jb_cleaned['days_rest'] = jb_cleaned['days_rest'].apply(lambda x: min(x, 30))

    # Calculate moving averages and other features
    jb_cleaned['over_target'] = (jb_cleaned['PTS'] > target_points).astype(int)
    jb_cleaned['avg_PTS_5'] = jb_cleaned['PTS'].rolling(window=5).mean().shift(1)
    jb_cleaned['avg_PTS_10'] = jb_cleaned['PTS'].rolling(window=10).mean().shift(1)
    jb_cleaned['avg_TPA_5'] = jb_cleaned['TPA'].rolling(window=5).mean().shift(1)
    jb_cleaned['trend_PTS'] = jb_cleaned['PTS'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True).shift(1)
    jb_cleaned['perc_over'] = jb_cleaned['over_target'].rolling(window=10).mean().shift(1) * 100

    jb_cleaned = jb_cleaned.dropna()

    # Select relevant variables for the model
    jb_cleaned['int_PTS5_TPA5'] = jb_cleaned['avg_PTS_5'] * jb_cleaned['avg_TPA_5']
    jb_cleaned['comp_opp_25_75'] = 0.25 * jb_cleaned['DEF_RTG'] + 0.75 * jb_cleaned['avgPTS_vteam']

    features = ['comp_opp_25_75', 'avg_PTS_5', 'trend_PTS', 'perc_over']

    X = jb_cleaned[features]
    y = jb_cleaned['over_target']

    # Fit logistic regression model
    X = sm.add_constant(X)
    glm_cleaned = sm.Logit(y, X).fit()

    # Add predicted probabilities to jb_cleaned
    jb_cleaned['predicted_prob'] = glm_cleaned.predict(X)
    jb_cleaned['predicted_class'] = (jb_cleaned['predicted_prob'] >= 0.5).astype(int)

    # Export the updated dataframe to CSV with dynamic filename
    filename = f"predictions_jb_{target_points}.csv"
    jb_cleaned[['Date', 'Opp', 'over_target', 'predicted_prob']].to_csv(filename, index=False)

    # Stratified cross-validation
    auc_scores = []
    accuracy_scores = []
    skf = StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(X, y):
        train, test = jb_cleaned.iloc[train_index], jb_cleaned.iloc[test_index]
        X_train = sm.add_constant(train[features])
        X_test = sm.add_constant(test[features])
        model_cv = sm.Logit(train['over_target'], X_train).fit(disp=0)
        predictions = model_cv.predict(X_test)
        predicted_classes = (predictions >= 0.5).astype(int)
        
        # Check if there are at least two classes in the target variable
        if len(test['over_target'].unique()) > 1:
            auc = roc_auc_score(test['over_target'], predictions)
            accuracy = accuracy_score(test['over_target'], predicted_classes)
            auc_scores.append(auc)
            accuracy_scores.append(accuracy)
        else:
            print(f"Skipping AUC calculation due to single class in the target variable.")
    
    if auc_scores:
        mean_auc = np.mean(auc_scores)
        mean_accuracy = np.mean(accuracy_scores)
        print(f'AUC: {mean_auc}')
        print(f'Accuracy: {mean_accuracy}')
    else:
        print("AUC calculation skipped for all iterations due to single class in the target variable.")

    # View updated dataset with predicted probabilities
    print(jb_cleaned[['Date', 'Opp', 'over_target', 'predicted_prob']].head(90))

# Run the model with a target of 21.5 points
# target = float(input("Enter your target: "))
# run_model_with_target(target)

# Run the model for a range of targets
list_of_targets = []

lb = float(input("Enter lower bound: "))
ub = float(input("Enter upper bound: "))

for i in range(int(ub - lb + 1)):
    list_of_targets.append(lb + i)

for i in range(len(list_of_targets)):
    run_model_with_target(list_of_targets[i])
