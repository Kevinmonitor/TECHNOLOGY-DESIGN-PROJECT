# TECHNOLOGY-DESIGN-PROJECT
Semester 1 2026.

## Data Architecture & Engineering Pipeline
*Owner: Bikram (Data Engineer)*

To ensure a true production-style NLP pipeline and prevent data leakage, we have centralized our data architecture. All text cleaning, standardization, and train/test splitting is now handled by a single Data Engineering script. 

**DO NOT write data cleaning or splitting code inside your individual modeling notebooks.**

### Repository Structure
* `data/raw/` - Contains the initial, unprocessed datasets.
* `data/processed/` - Contains the finalized `fpb_train.csv`, `fpb_val.csv`, `fpb_test.csv`, and `goemotions_5class.csv`. **Everyone must load their data from here.**
* `src/` - Contains our production Python scripts (e.g., `data_preprocessing.py`).
* `notebooks/` - All experimental Jupyter Notebooks (Baselines, BERT, LLMs) live here.
* `dashboard/` - The Streamlit app. Final evaluation metrics should be saved to `dashboard/data/` so the app can display them.

### How to Run the Data Pipeline.
If you are cloning this repository for the first time, or if the Data Engineer announces an update to the cleaning rules, generate your local data splits by running:
```bash
python src/data_preprocessing.py
