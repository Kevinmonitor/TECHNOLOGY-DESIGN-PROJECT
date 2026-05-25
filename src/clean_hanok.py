import json

nb_path = "notebooks/baseline_modelling.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_cells = []
for cell in nb["cells"]:
    source = "".join(cell.get("source", []))
    
    # Link Hanok's GoEmotions load to your new processed file
    if "goemotions = pd.read_csv" in source:
        cell["source"] = [s.replace("BASE + 'goemotions_5class.csv'", "BASE + '../data/processed/goemotions_5class.csv'") for s in cell["source"]]
        
    # Drop Hanok's redundant preprocessing cells entirely
    if "## 4. Text Preprocessing" in source: continue
    if "def preprocess_text" in source: continue
    if "### 4.1 Text Length After Preprocessing" in source: continue
    if "Word Count Before vs After Preprocessing" in source: continue
    
    new_cells.append(cell)

nb["cells"] = new_cells
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("[SUCCESS] Hanok's notebook surgically cleaned! Section 4 has been removed.")
