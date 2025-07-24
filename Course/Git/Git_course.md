
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

## Adding to the stating area

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


## Make a commit

git commit -m "adding the file file.md" (-m to include a log msg)
(Best pratice is to keep the log msg short a concise)

