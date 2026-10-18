# Materials Prediction Dashboard

This package turns your existing Colab models into a phone- and browser-friendly Streamlit web app. It does not retrain the models.

## 1. Export eight files from Colab
Run this in your current Colab notebook:

```python
from google.colab import files
from pathlib import Path
folder = Path("/content/cleaned_materials_data")
names = [
    "yield_strength_random_forest.joblib",
    "gamma_prime_radius_model.joblib",
    "gamma_prime_volume_fraction_model.joblib",
    "gamma_prime_spacing_model.joblib",
    "yield_strength_model_features.txt",
    "gamma_prime_model_metadata.json",
    "yield_strength_final_training_rows.csv",
    "gamma_prime_simulated_cleaned.csv",
]
for name in names:
    files.download(str(folder / name))
```

## 2. Create a GitHub repository
Create a repository such as `materials-prediction-dashboard`. Upload `app.py`, `requirements.txt`, and the eight downloaded files to the repository root. Keep filenames exactly the same.

## 3. Deploy
1. Visit https://share.streamlit.io/ and sign in with GitHub.
2. Choose **Create app**.
3. Select your repository, branch, and `app.py`.
4. Deploy. Streamlit will give you a web URL.
5. Open the URL in Chrome on your phone or any browser.

## Important
- This starter ZIP does not contain your model files because they are in your Colab runtime. You must export and upload the eight files above.
- Only upload model files you created and trust; loading arbitrary joblib files can execute unsafe code.
- γ′ models use synthetic illustrative training data and have weak test performance. They are not experimentally validated.
- Yield strength uses median recorded composition for the selected alloy. γ′ uses separate composition inputs, so this is not a coupled simulation.
