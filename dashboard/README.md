# Sentiment Analysis Dashboard — COS60011

Interactive dashboard for the project demo (Microsoft Teams live presentation).

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy your data files into dashboard/data/
#    Required now:
#      - goemotions_5class.csv
#      - fpb_train.csv
#      - fpb_val.csv
#      - fpb_test.csv
#      - task2_baseline_results.csv  (from Hanok's notebook)
#
#    Add when ready:
#      - bert_results.csv           (Aniketh)
#      - model_c_results.csv        (Fin)
#      - llm_results.csv            (Kevin)

# 3. Run the dashboard
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

## For the Teams Demo

1. One person runs the dashboard locally (Bikram recommended — he built the data pipeline)
2. Share screen on Teams
3. Each team member navigates to their tab when it's their turn
4. Use the interactive filters/dropdowns during Q&A to answer panel questions

## Adding Your Results

Each team member exports their results as a CSV into `dashboard/data/`.

### Aniketh — bert_results.csv
```
model,accuracy,f1_macro,f1_weighted,precision_macro,recall_macro,f1_fear,f1_joy,f1_neutral,f1_optimism,f1_sadness
Model A,0.65,0.45,0.62,0.50,0.42,0.10,0.15,0.78,0.55,0.35
Model B,0.80,0.68,0.78,0.70,0.65,0.30,0.40,0.88,0.72,0.55
```

### Fin — model_c_results.csv
```
stage,accuracy,f1_macro,f1_weighted,f1_fear,f1_joy,f1_neutral,f1_optimism,f1_sadness
After GoEmotions,0.60,0.40,0.55,0.05,0.10,0.75,0.50,0.30
After FPB fine-tune,0.82,0.72,0.80,0.35,0.45,0.90,0.75,0.60
```

### Kevin — llm_results.csv
```
model,strategy,accuracy,f1_macro,f1_weighted,f1_fear,f1_joy,f1_neutral,f1_optimism,f1_sadness
Gemma 3 4B,zero-shot,0.45,0.30,0.42,0.05,0.08,0.65,0.35,0.20
Gemma 3 4B,few-shot (5),0.52,0.38,0.50,0.10,0.15,0.70,0.42,0.28
Llama 3.2 3B,zero-shot,0.42,0.28,0.40,0.03,0.06,0.62,0.32,0.18
Llama 3.2 3B,few-shot (5),0.49,0.35,0.47,0.08,0.12,0.68,0.40,0.25
```

## File Structure

```
dashboard/
├── app.py                  # Main entry point + sidebar + routing + CSS
├── utils.py                # Shared constants, plotly theme, helpers
├── requirements.txt
├── README.md
├── data/                   # All CSV files go here
│   ├── goemotions_5class.csv
│   ├── fpb_train.csv
│   ├── fpb_val.csv
│   ├── fpb_test.csv
│   ├── task2_baseline_results.csv
│   ├── bert_results.csv        (when ready)
│   ├── model_c_results.csv     (when ready)
│   └── llm_results.csv         (when ready)
└── pages/
    ├── __init__.py
    ├── overview.py             # Tab 1: Fin
    ├── data_pipeline.py        # Tab 2: Bikram
    ├── baselines.py            # Tab 3: Hanok
    ├── bert_finetuning.py      # Tab 4: Aniketh
    ├── sequential_transfer.py  # Tab 5: Fin + Bikram
    └── llm_experiments.py      # Tab 6: Kevin
```

## Himanshu's Comparison Tab

Once all results are in, uncomment the "Model Comparison" tab in `app.py` 
and create `pages/comparison.py` to pull from all CSVs and render 
side-by-side comparisons across all approaches.
