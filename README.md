# Cricket Match Winner Prediction for the World Cup 2019

## 📌 Problem Statement
Predict the winner of cricket matches for the World Cup 2019 using team identities and ICC rankings, then simulate league, semi-final, and final outcomes.

---

## 📁 Dataset & Data Handling
* **`results.csv`**: Past international match results with winner labels.
* **`fixtures.csv`**: Scheduled matches in the tournament.
* **`icc_rankings.csv`**: Team rankings used to add positional features.

---

## 🛠 Preprocessing & Approach
* **Preprocessing**:
  * Filter to World Cup teams.
  * Drop irrelevant columns (`date`, `Margin`, `Ground`).
  * One-hot encode `Team_1` and `Team_2` into binary features.
* **Approach**:
  * Train/test split using `train_test_split` (70/30).
  * Logistic Regression as baseline classifier.
  * Generate predictions for league stage, semi-finals, and finals via `predict_result` function.

---

## 📊 Results
* Training accuracy and testing accuracy are printed by the script.
* *Note: Training accuracy higher than testing accuracy suggests some overfitting due to small dataset size and simple model.*

---

## 🚀 How to Run

### Local Setup
```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run prediction script
cd src
python prediction.py

## Traning accuracy:  0.7216
## Testing accuracy:  0.587360594795539

![image](https://user-images.githubusercontent.com/57340784/206966014-2cef8d87-ec77-4f47-8e52-3bcc4d4e6350.png)

![image](https://user-images.githubusercontent.com/57340784/205433867-205b110c-e00f-4fbb-ad43-48ef7efd4fd7.png)

