import json

nb_path = "notebooks/baseline_modelling.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        new_source = []
        for line in cell.get("source", []):
            # 1. Fix LogisticRegression (Both make_pipelines and learning_curve)
            line = line.replace("solver='lbfgs'", "solver='lbfgs', random_state=SEED")
            
            # 2. Fix LinearSVC in make_pipelines (handles the newline format)
            line = line.replace("class_weight='balanced'\n", "class_weight='balanced', random_state=SEED\n")
            
            # 3. Fix LinearSVC in learning_curve (handles the closing parenthesis format)
            line = line.replace("class_weight='balanced'))", "class_weight='balanced', random_state=SEED))")
            
            new_source.append(line)
        cell["source"] = new_source

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("[SUCCESS] Explicit random_state=SEED added to all LogisticRegression and LinearSVC definitions!")
