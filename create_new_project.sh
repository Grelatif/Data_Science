#!/bin/bash

# ce programme permet de creer l'arborescence de projets automatiquement.
# Il demande le themre et le nom du projet et en avant!

echo "Thème du projet ? (ex: Temporal_Series, NLP...)"
read THEME

echo "Nom du projet ? (ex: Store_Sales_Forecast)"
read NAME

BASE="Projects/$THEME/$NAME"

mkdir -p "$BASE"
cd "$BASE" || exit

# Créer les fichiers
touch requirements.txt
touch notebook.ipynb

# Générer .gitignore
echo -e ".venv/\n__pycache__/\n.ipynb_checkpoints/\n.env" > .gitignore

# Générer README
cat <<EOT > README.md
# $NAME

## 📌 Contexte
...

## 📊 Données
...

## ⚙️ Méthodologie
...

## 🧪 Reproduire l'environnement

\`\`\`bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
\`\`\`

EOT

echo "✅ Projet créé dans $BASE"
