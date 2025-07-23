# Guide sur Git et Environnements Virtuels

## Objectif
Documenter l'utilisation de Git et des environnements virtuels (`venv`) pour gérer mes projets dans Data_Science.
- **Intérêt** : Fournir une référence centralisée pour relire et comprendre les outils utilisés, évitant de perdre du temps à réapprendre.

## Git
- **git init** : Initialise un dépôt Git local (utile pour commencer à suivre les changements).
- **git add <fichier>** : Ajoute un fichier aux modifications à commiter (prépare les changements pour validation).
- **git commit -m "message"** : Valide les modifications avec un message (enregistre une étape dans l'historique).
- **git push origin main** : Envoie les modifications au dépôt distant (partage les changements avec d'autres ou sauvegarde en ligne).
- **git log <fichier>** : Affiche l'historique des commits pour un fichier (permet de retracer les modifications).
- Déplacer un fichier : `mv ancien_chemin nouveau_chemin`, puis `git add` et `git commit` (gère les déplacements dans le suivi Git).
- **Intérêt** : Permet de suivre l'évolution des fichiers, collaborer avec d'autres, et sauvegarder les changements de manière organisée.

## Environnements Virtuels (venv)
- **python -m venv venv** : Crée un environnement virtuel dans un dossier `venv` (isolé pour chaque projet).
- **source venv/bin/activate** (Linux/Mac) ou `venv\Scripts\activate` (Windows) : Active l'environnement (charge les dépendances spécifiques).
- **pip install <package>** : Installe une dépendance dans le `venv` (limite l'installation à cet environnement).
- **pip freeze > requirements.txt** : Génère un fichier de dépendances (facilite la reproduction de l'environnement).
- **deactivate** : Désactive l'environnement (revient à l'environnement global).
- **Intérêt** : Évite d'installer plein de librairies globalement, résout les conflits de versions, et garantit une reproductibilité propre sur différentes machines.

## Gestion Multi-Projets
- Chaque sous-dossier (ex. `Exercices`) peut avoir son propre `requirements.txt` et `venv` (permet une gestion indépendante des dépendances).
- Placer les fichiers à la racine de chaque sous-projet (facilite l'organisation et l'accès).
- **Intérêt** : Permet de gérer plusieurs projets indépendamment sans mélange de dépendances, facilitant la maintenance et l'évolutivité.

# Bonne pratiques 

## Structure Projet

```
mon-projet/
│
├── data/
│   ├── raw/          # Données brutes (non versionnées)
│   ├── processed/    # Données nettoyées (non versionnées)
│   └── sample/       # Échantillons légers pour test (versionnés si petit)
│
├── notebooks/        # Jupyter notebooks
├── src/              # Code source Python (modules, fonctions)
├── models/           # Modèles enregistrés (non versionnés)
├── outputs/          # Graphiques, résultats (optionnel)
├── requirements.txt  # Dépendances Python
├── environment.yml   # (si tu utilises conda)
├── README.md
├── .gitignore
└── .env              # Variables d’environnement (jamais push sur GitHub)
```

## Notebooks 
- Bien documenter les notebooks.
- Utiliser un format clair : 1 notebook = 1 étape (exploration, modélisation, évaluation…)

## README
Inclure :
- Objectif du projet
- Données utilisées (et où les trouver)
- Instructions pour exécuter le code
- Organisation du projet
