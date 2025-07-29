
## What is a Git repo?

DIrectory containing files, subdirectories AND a Git storage (.git) that stores information about the project history. Never edit the .git

## What are the benefits?

Track, revert, changes, collaboration with collegues.

## Creating a new repo

git init "project_name"

cd project_name
git status

## Convert and existing project to a repo

git init (from the project location)

## Note

Avoid creating nested git repos. (there will be 2 .git directories).

## Git Workflow

1/ Edit and save files on our computer

2/ Add files to the Git staging area (place the letter in the enveloppe)
Track what has been modified

3/ Commit the files (putting the enveloppe in the mailbox)
Git takes a snapshot of the files at that point in time (allow to compare files, revert previous versions)


We can put/change whatever we want in the enveloppe, but when we post it to the mailbox, we can't change it.

## Adding to the stating area(index)

1/ add a single file:
git add file.md

2/ add all modified files
git add . (all files of the current directory and subdirectoriess)


When we add files with git add, we place modified, new, or deleted files in the staging area (an intermediate zone). If the changes look good, we commit them to the current branch (e.g., main) and can then push the commits to the remote repository."

Points clés sur la staging area
La staging area (ou index) est une étape intermédiaire entre ton répertoire de travail (working directory) et l’historique des commits.
Elle te permet de préparer un commit en sélectionnant précisément les modifications à inclure (par exemple, tu peux ajouter seulement certaines lignes d’un fichier avec git add -p).
Elle est spécifique à la branche sur laquelle tu travailles (par exemple, main dans ton cas).
Une fois les modifications mises en staging, tu les valides avec git commit, ce qui les enregistre dans la branche locale. Le git push intervient ensuite pour synchroniser avec le dépôt distant.

On peut également supprimer un fichier avec un "add" (contre intuitif)

Si j'ai add un fichier et que je le modifie ensuite, il faut le add une nouvelle fois pour que le fichier modifié soit dans la stagin area.

## Make a commit

git commit -m "adding the file file.md" (-m to include a log msg)
(Best pratice is to keep the log msg short a concise)

Apres un commit, les fichiers de la stagin area, restent.

A chaque commit sont associés l'heure/date, l'author, l'ID du commit.

Chaque commit contient un snapshot complet du projet.

Ne pas faire de commit pour de tout petits changements ou au contraire pour d'énormes modifs.
Idéalement, un commit pour une tache spécifique (bug fix par exemple)
Utiliser le présent (Fix the bug) plutot que le passé (fixed the bug)

Il est possible de commit sans passer par la staging area, mais c'est risqué et a faire suelement si on est certain de ce qu'on fait. 

Si je modifie de nom d'un fichier de la staging area (avec mv oldfile newfile), il faudra add oldfile pour delete et add newfile pour ajouter a la staging area (ca equivaut a supprimer un fichier et en créer un nouvreau).
Cependant si j'utilise "git mv" au lieu de "mv", alors git comprend que c'est un rename le changement s'applique à la fois sur le working directory et la staging area (donc paas besoin de re-add).


## Gitignore

Le fichier **`.gitignore`** dans Git est utilisé pour indiquer à Git quels fichiers ou dossiers il doit **ignorer** lorsqu'il suit les modifications dans un dépôt. Cela signifie que les fichiers ou dossiers listés dans `.gitignore` ne seront pas ajoutés à la **staging area** (avec `git add`) ni inclus dans les commits, même s'ils sont modifiés ou présents dans le répertoire de travail. Ce mécanisme est utile pour exclure des fichiers temporaires, des fichiers générés automatiquement, des données sensibles ou des fichiers qui n’ont pas besoin d’être versionnés.

### Concept clé du `.gitignore`
- **But** : Éviter que des fichiers inutiles ou sensibles (comme des fichiers de configuration locaux, des fichiers de compilation, ou des clés API) soient suivis par Git et envoyés au dépôt distant (par exemple, sur GitHub).
- **Emplacement** : Le fichier `.gitignore` est généralement placé à la racine du dépôt Git, mais il peut aussi exister dans des sous-dossiers pour des règles spécifiques.
- **Syntaxe** : Les règles dans `.gitignore` utilisent des motifs (patterns) pour spécifier les fichiers ou dossiers à ignorer. Ces motifs peuvent inclure des noms de fichiers, des extensions, ou des expressions génériques.

### Règles courantes dans `.gitignore`
- **Fichiers spécifiques** : Par exemple, `secret.txt` ignore un fichier nommé `secret.txt`.
- **Extensions** : Par exemple, `*.log` ignore tous les fichiers avec l’extension `.log`.
- **Dossiers** : Par exemple, `node_modules/` ignore le dossier `node_modules` et tout son contenu.
- **Motifs génériques** : `*` pour tout caractère, `**` pour inclure les sous-dossiers.
- **Exceptions** : Un `!` permet d’inclure un fichier ou dossier qui serait autrement ignoré. Par exemple, `*.log` suivi de `!important.log` ignore tous les fichiers `.log` sauf `important.log`.

### Exemple simple dans GitHub Codespaces
Imaginons que vous travaillez dans un dépôt dans Codespaces, et votre projet contient :
- Un fichier `README.md` (que vous voulez versionner).
- Un fichier `notes.txt` (fichier personnel que vous ne voulez pas versionner).
- Un dossier `temp/` contenant des fichiers temporaires (à ignorer).
- Des fichiers `.log` générés par une application (à ignorer).

Voici comment utiliser `.gitignore` pour gérer cela :

1. **Créer un fichier `.gitignore`** :
   Dans le terminal de Codespaces:
   ```bash
   touch .gitignore
   ```

2. **Éditer `.gitignore`** :
   Ouvrez `.gitignore` (par exemple, avec `nano .gitignore` ou dans l’éditeur VS Code) et ajoutez :
   ```gitignore
   # Ignorer le fichier notes.txt
   notes.txt

   # Ignorer le dossier temp et tout son contenu
   temp/

   # Ignorer tous les fichiers .log
   *.log
   ```

3. **Vérifier l’effet** :
   - Modifiez `notes.txt`, créez un dossier `temp/test.txt`, ou ajoutez un fichier `app.log`.
   - Exécutez :
     ```bash
     git status
     ```
     Vous verrez que `notes.txt`, `temp/`, et `app.log` **n’apparaissent pas** dans la liste des fichiers à ajouter, car ils sont ignorés par `.gitignore`. En revanche, si vous modifiez `README.md`, il apparaîtra dans `git status`.

4. **Valider `.gitignore`** :
   Le fichier `.gitignore` lui-même doit être versionné pour s’appliquer à tous les collaborateurs du dépôt :
   ```bash
   git add .gitignore
   git commit -m "Add .gitignore to ignore notes.txt, temp/, and .log files"
   git push origin main
   ```

### Ce qui se passe sans `.gitignore`
Si vous n’utilisez pas `.gitignore` :
- Tous les fichiers modifiés ou nouveaux (comme `notes.txt`, `temp/test.txt`, ou `app.log`) apparaîtront dans `git status` comme "untracked files" ou "changes not staged".
- Vous risquez d’ajouter accidentellement ces fichiers avec `git add .` et de les inclure dans un commit, ce qui peut encombrer le dépôt ou exposer des données sensibles.


## Check the modifications made on the staging files 
L'idée est de comparer ce qu'on a changé entre le dernier commit (ref) et la staging area actuelle.
Pour ce faire on utilise la commande 
```bash
git diff --staged
```

Cela nous mettra en évidence les lignes ajoutées/supprimées dans les fichiers de la staging area (par rapport au fichier du dernier commit)
Si on utilise "git diff" sans le --staged, on compare la staging area (ref) avec le working directory. 

## Logs
git log permet de :
- Suivre l’historique : Consulter la chronologie des modifications apportées au dépôt (par exemple, dans GitHub Codespaces, pour voir les commits sur votre projet).
- Débogage : Identifier quand une modification particulière (par exemple, une correction de "typo" ou un bug) a été introduite.
- Collaboration : Vérifier qui a effectué des changements et à quel moment.
- Navigation : Trouver un commit spécifique pour y revenir (avec git checkout, git revert, etc.).
- Audit : Examiner les messages de commit pour comprendre pourquoi des changements ont été faits.

git show "ID" pour voir en détails un commit particulier, via son unique ID.
git ls -tree pour l'ensemble du snapshot d'un commit

## Unstaging files
On pourrait vouloir unstage des fichiers (par exemple parce qu'on pense que les modifs devraient faire partie d'un autre commit, car partie d'une autre tache que celle associée à ce commit.)
Pour ce faire on va utiliser la commande git restore.
Si on veut unstage le file1.txt de la staging area: 
```bash
git restore --staged file1.txt
```
(on peut utilsier des patterns avec * ou prendre tous les fichiers avec .)

## Discard local changes
On utilise git clean pour remove des fichiers "untracked" (pas dans la staging area et pas de commit)
git clean est utilisé pour:
- Nettoyer le répertoire de travail : Supprime les fichiers et dossiers non suivis qui ne sont pas nécessaires pour le projet (par exemple, fichiers temporaires, fichiers de compilation, ou fichiers créés par erreur).
- Préparer un dépôt propre : Utile avant un commit, un push, ou un changement de branche pour éviter que des fichiers non suivis ne causent de confusion.
- Complément à .gitignore : Si des fichiers non suivis auraient dû être ignorés par .gitignore, git clean peut les supprimer en masse après avoir mis à jour .gitignore.
- Points importants
- Fichiers affectés : git clean ne touche que les fichiers non suivis (ceux listés sous "Untracked files" dans git status). Les fichiers suivis (tracked) ou ceux dans la staging area ne sont pas affectés.
- Irréversible : Les fichiers supprimés par git clean ne peuvent pas être récupérés via Git, car ils n’étaient pas versionnés. Soyez prudent !
- Utilisation avec précaution : Il est recommandé de vérifier ce qui sera supprimé avant d’exécuter la commande


git rm file remove le fichier a la fois de la staging area et du working directory

git clean supprime les fichiers non suivis (untracked), tandis que git rm s’applique aux fichiers suivis (tracked).

## restore a file from a previous version
cela sert à :
- Annuler les modifications dans le répertoire de travail : Restaure les fichiers modifiés dans leur état tel qu’il était dans le dernier commit, supprimant ainsi les modifications non ajoutées.
- Retirer des fichiers de la staging area : Annule l’ajout de modifications à la staging area (équivalent à défaire un git add), tout en préservant les modifications dans le répertoire de travail.
- Remplacement partiel de commandes obsolètes : Depuis Git 2.23 (2019), git restore remplace en partie git checkout pour certaines opérations de restauration, offrant une syntaxe plus claire.

On peut restore par rapport a l'ID d'un ancien commit:
```bash
git restore --source=ghi7890 --staged README.md
```