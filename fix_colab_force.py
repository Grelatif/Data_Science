import json, subprocess
from pathlib import Path

OWNER = "ElMartinez31"
REPO = "Data_Science"
BRANCH = "main"
BADGE_IMG = "https://colab.research.google.com/assets/colab-badge.svg"

# Liste des notebooks suivis par git
nb_paths = subprocess.check_output(
    ["git", "ls-files", "*.ipynb"], text=True
).splitlines()

def make_badge(path):
    return f"[![Open In Colab]({BADGE_IMG})](https://colab.research.google.com/github/{OWNER}/{REPO}/blob/{BRANCH}/{path})"

changed_files = []

for nb_path in nb_paths:
    rel_path = Path(nb_path).as_posix()
    badge_md = make_badge(rel_path)

    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        print(f"⚠️ Impossible de lire {nb_path} : {e}")
        continue

    updated = False
    badge_done = False

    # Vérifie chaque cellule Markdown existante
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            source = "".join(cell.get("source", []))
            if "colab.research.google.com/github" in source:
                if badge_md not in source:
                    cell["source"] = [badge_md]
                    updated = True
                badge_done = True
                break

    # Si aucun badge trouvé, on l'ajoute en première cellule
    if not badge_done:
        new_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [badge_md]
        }
        nb["cells"].insert(0, new_cell)
        updated = True

    if updated:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False)
        changed_files.append(nb_path)

print("\n✅ Badges forcés/corrigés dans :")
for f in changed_files:
    print("-", f)
print(f"\nTotal corrigés : {len(changed_files)}")
