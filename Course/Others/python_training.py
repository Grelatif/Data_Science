######################################### Training Python 2024 session ##############################################


						######## Set Up ###############

# Conda installing and settting up: https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html
# Jupyter Install : https://jupyter.org/install

conda create -n py39 python=3.9 # In order to create env with updated python version : 

conda info --envs
conda activate <env>
python --version  # to get the version of pyton
# Data Science packages: https://www.simplilearn.com/top-python-libraries-for-data-science-article

conda activate py39 # In order to activate and use this env

conda install TensorFlow # to install a package

						######## Websites ###############
#https://www.w3schools.com/python/
#https://www.kaggle.com/ #my_filepath = "../input/fish-csv/fish_data.csv" to retrieve csv in a notebook
#https://medium.com/@gubrani.sanya2/data-manipulation-with-pandas-399213045b91
#http://www.python-simple.com/python-pandas/creation-series.php

						######## Python Data types ###############
#By default Python have these data types:

#strings - used to represent text data, the text is given under quote marks. e.g. "ABCD"
#integer - used to represent integer numbers. e.g. -1, -2, -3
#float - used to represent real numbers. e.g. 1.2, 42.42
#boolean - used to represent True or False.
#complex - used to represent complex numbers. e.g. 1.0 + 2.0j, 1.5 + 2.5j

						######## Numpy training ###############

##### Data Types in Numpy
#NumPy has some extra data types, and refer to data types with one character, like i for integers, u for unsigned integers etc.

#Below is a list of all data types in NumPy and the characters used to represent them.

i - integer
b - boolean
u - unsigned integer
f - float
c - complex float
m - timedelta
M - datetime
O - object
S - string
U - unicode string
V - fixed chunk of memory for other type ( void )




##### Arrays

#Les tableaux (en anglais, array) peuvent être créés avec numpy.array(). On utilise des crochets pour délimiter les listes d’éléments dans les tableaux.
a = np.array([1, 2, 3, 4]) # Utilisation des brackets et les éléments sont séparés par les ","
>>a
array([1, 2, 3, 4])
# Acces aux elements:
a[0] #First index starts at 0!
1
a[3] # And ends at n-1
4

#2D arrays:
b = np.array([[1, 2, 3], [4, 5, 6]]) # Brackets imbriqués # this gives a 2 rows and 3 columnss array 
# Acces aux éléments
b[0,1] # first arg is which row and second is which column, here first row, second column
2
b[1,2]
6
arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print('2nd element on 1st row: ', arr[0, 1]) # how to properly print it
print('Last element from 2nd row: ', arr[1, -1])#Use negative indexing to access an array from the end.

##### Access to several values at once:

c = np.array([[1, 2, 3,123], [4, 5, 6,456],[7,8,9,789]])
>>array([[  1,   2,   3, 123],
       [  4,   5,   6, 456],	
       [  7,   8,   9, 789]])
c[2,:] # gives all values of the third row array([  7,   8,   9, 789])
c[:,0] # gives all values of first column array([1, 4, 7])
c[2,1:4] # gives values between index 1 (INCLUDED!) until index 4 (EXCLUDED!) of the third row >> array([  8,   9, 789])
c[1:,3] # gives all rows (starting at the second one) until the end, for column number 4 >> array([456, 789])


##### Useful functions:

m = np.arange(3, 15, 2) # allow to create an array from 3 to 15 with a step of 2
m
array([ 3,  5,  7,  9, 11, 13])

#numpy.linspace() permet d’obtenir un tableau 1D allant d’une valeur de départ à une valeur de fin avec un nombre donné d’éléments.
np.linspace(3, 9, 10)
array([ 3.        ,  3.66666667,  4.33333333,  5.        ,  5.66666667,
        6.33333333,  7.        ,  7.66666667,  8.33333333,  9.        ])

numpy.around(x,n) #arrondi à n décimales
numpy.trunc(x) #retourne la partie entière du nombre (le nombre est tronqué)
numpy.random.random() #permet d’obtenir des nombres compris entre 0 et 1 par tirage aléatoire avec une loi uniforme

##### Create array with a specific datatype
arr = np.array([1, 2, 3, 4], dtype='S') #to get strings format
arr = np.array(['a', '2', '3.2145'], dtype='i') # to get it as integer, but raises an error cause of the "a"


results = {} ###### Inititaliser une liste vide, tres utile !!!!!!!!!!!!!!!!!!!!!!!!!!!!
for i in range(1,9):
    results[50*i] = get_score(50*i)
    plt.plot(list(results.keys()), list(results.values())) ####to visualize the results for each n_estimators
plt.show()


						######## Pandas training ###############
##### Read external data:

 # Importing data from a CSV file
df = pd.read_csv('file.csv', , index_col="index_col")
stocks = pd.read_csv("../input/nyse/prices.csv", parse_dates=['date']) #use of parse dates
stocks = stocks[stocks['symbol'] == "GOOG"].set_index('date') # set index
# Importing data from an Excel file
df = pd.read_excel('file.xlsx')

# Exporting data to a CSV file
df.to_csv('new_file.csv')

# Exporting data to an Excel file
df.to_excel('new_file.xlsx')

#http://www.python-simple.com/python-pandas/lecture-ecriture-fichier.php   pour plus d'info

##### Data structures: 

Axis 0 will act on all the ROWS in each COLUMN
Axis 1 will act on all the COLUMNS in each ROW
+------------+---------+--------+
|            |  A      |  B     |
+------------+---------+---------
|      0     | 0.626386| 1.52325|----axis=1----->
+------------+---------+--------+
             |         |
             | axis=0  |
             ↓         ↓



# Series 
#A Series is a one-dimensional labeled array
student_scores = pd.Series([85, 92, 78, 95])
a = pandas.Series([1, 2, 5, 7], dtype = float) #Type added
s = pandas.Series(['a', 'b', 'a', 'a', 'c'], dtype = 'category')
len(s) or s.size #taille de la série
[s < 5] = 99 # met à 99 les valeurs inférieures à 5
s[numpy.isnan(s)] = 99 # met à 99 les valeurs nan (attention : s[s == numpy.nan] = 99 ne marche pas, car numpy.nan == numpy.nan est False !)
s2 = s.copy() # fait une copie indépendante de la série.
#Modification de valeurs :
a.replace(2,20) # remplace les occurences de 2 par 20 dans la série a
s.replace('a','A') # remplace les 'a' par 'A'
s.replace({'a': 'A', 'b': 'B'}) : remplace les 'a' par des 'A' et 'b' par des 'B' et renvoie une série modifiée.
s.map({'a': 'A', 'b': 'B'}), mais les valeurs non présentes dans le dictionaire seront mises à NaN
s.map(lambda x: 'A' if x == 'a' else 'B' if x == 'b' else numpy.nan)
s.map(lambda x:str.upper(x)) # is smarter and applies for all

#Index d'une série : c'est le nom affecté à chaque valeur :
pandas.Series([1, 2, 5, 7], index = ['a', 'b', 'c', 'd']) : permet de donner des noms aux individus (i.e. à chaque valeur).
#pour donner des noms, on peut aussi utiliser un dictionnaire : pandas.Series({'a': 1, 'b': 2, 'c': 5, 'd': 7}) revient au même que l'exemple juste au-dessus (si on donne un dictionnaire, les éléments sont triés par ordre alphabétique des clefs).
mySerie.index # renvoie l'index d'une série qui est de la classe Index.
#on peut réindexer une série : dans le cas précédent, s.reindex(['c', 'b', 'a', 'e']) renvoie la série (5, 2, 1, NaN) (car par de valeur associée à l'index 'e').


# DataFrame:
#A DataFrame is a two-dimensional labeled data structure resembling a spreadsheet
# Create a dictionary with student data
student_data = {
    "Name": ["Alex", "Ben", "Clara", "Deric", "Eva", "Frank"],
    "Age": [20, 21, 19, 22, 23, 20],
    "Major": ["CS", "Math", "Physics", "Economics", "Biology", "Chemistry"],
    "Score": [85, 92, 78, 95, 88, 90],
    "Hometown": ["York", "Angeles", "Chicago", "Houston", "Boston", "Francisco"],
    "Graduation Year": [2022, 2023, 2022, 2023, 2024, 2023],
    "Scholarship": [False, True, False, True, False, False]
}

# Note that a dictionary is a list of couples <key : value>, where Key would be the column and values will be the entries
# Create a DataFrame from the dictionary
df = pd.DataFrame(student_data)


#un dataframe se comporte comme un dictionnaire dont les clefs sont les noms des colonnes et les valeurs sont des séries (donc les valeurs d'une série correspondent à une colonne).
#on peut le créer à partir d'une array numpy (mais ce n'est pas très pratique et le type des données est le même pour toutes les colonnes, ici float64) 
ar = numpy.array([[1.1, 2, 3.3, 4], [2.7, 10, 5.4, 7], [5.3, 9, 1.5, 15]])
df = pandas.DataFrame(ar, index = ['a1', 'a2', 'a3'], columns = ['A', 'B', 'C', 'D']) 
# Creation comme un seul dico
df = pandas.DataFrame({'A': [1.1, 2.7, 5.3], 'B': [2, 10, 9], 'C': [3.3, 5.4, 1.5], 'D': [4, 7, 15]},
                      index = ['a1', 'a2', 'a3'])
#on peut aussi donner une liste de dictionnaires :
pandas.DataFrame([{'A': 1.1, 'B': 2, 'C': 3.3, 'D': 4},
    {'A': 2.7, 'B': 10, 'C': 5.4, 'D': 7},
    {'A': 5.3, 'B': 9, 'C': 1.5, 'D': 15}]) #est équivalent


### Index , LOC and ILOC
#Index is like an address, that’s how any data point across the dataframe or series can be accessed. Rows and columns both have indexes, rows indices are called as index and for columns 
#its general column names.
df.loc #are for labels/ names
df.iloc# are for position numbers
#un index ou les colonnes d'un dataframe peuvent avoir un nom :
#df.index.name = 'myIndexName' (si on imprime le frame dans un fichier csv avec l'index, la colonne sera nommée avec le nom de l'index).
#df.rename(columns = {'A':'new_name'}, inplace = True)
#Autres initialisations de dataframes :
#df = pandas.DataFrame(columns = ['A', 'B']) : dataframe avec 0 lignes
#df = pandas.DataFrame(columns = ['A', 'B'], index = ['a', 'b']) : dataframe avec 2 lignes et que des NA
#df = pandas.DataFrame(0, index = [0, 1], columns = ['a', 'b']) : dataframe initialisé avec que des 0.
#df.fillna(0, inplace = True) : le remplit avec des 0 plutot que des NaN
#mais, attention ! : initialement, les types des colonnes sont "object" et une colonne peut avoir des valeurs de types héterogènes !
#pour éviter ça, on peut donner un type à la création : df = pandas.DataFrame(columns = ['A', 'B'], index = ['a', 'b'], dtype = float) (ou numpy.float64, ça revient au même)
#on peut réindexer un dataframe pour changer l'ordre des lignes et/ou des colonnes, ou n'en récupérer que certaines : df.reindex(columns = ['C', 'B', 'A'], index = ['a2', 'a3'])
#df.dtypes : les types des différentes colonnes du dataframe, ici :

# usuefull commands
#!!!!! Both loc and iloc are row-first, column-second. This is the opposite of what we do in native Python, which is column-first, row-second!!!!!
#!!!!! ILOC uses the Python stdlib indexing scheme, where the first element of the range is included and the last one excluded. So 0:10 will select entries 0,...,9. !!!!!
#!!!!! LOC, meanwhile, indexes inclusively. So 0:10 will select entries 0,...,10. !!!!!
#This is particularly confusing when the DataFrame index is a simple numerical list, e.g. 0,...,1000. In this case df.iloc[0:1000] will return 1000 entries
# while df.loc[0:1000] return 1001 of them! To get 1000 elements using loc, you will need to go one lower and ask for df.loc[0:999].
df.index # nous donne l'index
df.columns
df.values
df['A'] # renvoie la Series correspondant à la colonne de label A :
df.loc[:, 'A'] : renvoie une series
df.loc[:, ['A']] : renvoie un dataframe.
df.iloc[3, :] : renvoie une series.
df['A'][0:3] : les 3 premières valeurs des 3 premières lignes de la colonne 'A' (sous forme de Series)
df[['A'][0:3]] meme chose sous forme de dataframe
df.loc['a2'] # renvoie la Series correspondant à la ligne d'index a2 
df.loc[['a2', 'a3'], ['A', 'C']] : renvoie un dataframe avec un sous-ensemble des lignes et des colonnes :
df.loc[:,['A', 'C']] : toutes les lignes et seulement les colonnes A et B.
df.loc['a2', 'C'] : accès à la valeur de la ligne a2 et de la colonne C : 5.4.
df.at['a2', 'C'] # autre façon recommandée d'accéder à la valeur de la ligne a2 et de la colonne C : 5.4.
df.iloc[1] : renvoie la deuxième ligne.
df.iloc[1:3,[0, 2]] : renvoie le dataframe avec les lignes 1 à 3 exclue, et les colonnes numéros 0 et 2.
df.iloc[:,2:4] : renvoie toutes les lignes et les colonnes 2 à 4 exclue.
df.iloc[1,2] : renvoie la valeur à la ligne 2 et la colonne 3.
on peut aussi faire une affectation pour changer la valeur : df.at['a2', 'C'] = 7.
df.loc[:,['A']] est un dataframe (avec une seule colonne).
df.loc[:,'A'] est une series, comme df['A'].
df.loc[df.index[3], 'A']
ou alors df.iloc[3, df.columns.get_loc('A')] # mélange de loc et iloc
reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)] # Loc sert de filtre sur les pays qui sont italy ou sur les points revus > 90
reviews.loc[reviews.country.isin(['Italy', 'France'])] # meme idée avec isin
df[df['A'] > 2] : renvoie un dataframe avec seulement les lignes où la condition est vérifiée :
df['myCol'].dtype # donne le type de la colonne
df.shape # renvoie la dimension du dataframe sous forme (nombre de lignes, nombre de colonnes)
len(df) ou len(df.index) # renvoie le nombre d'entrées (lignes)
len(df.columns) # nb de colonnes
df.memory_usage() # mémoire utilisée
Pour compter le nombre de lignes pour lesquelles on a une valeur : (df['A'] == 'x').sum()
#Sampling d'un dataframe :
df.sample(frac = 0.1) : retourne un dataframe avec 10% des lignes de départ, sans remise.
df.sample(frac = 1) : fait un shuffling des lignes.
df.sample(frac = 0.5, axis = 1): retourne 50% des colonnes.

#e.g. Lets assume Ram, Sonu & Tony are standing at positions 1, 2 & 3 respectively. If you want to call Ram you have two options, either you call him by his name or his position number. So, if you call Ram by his name “Ram”, you will use df.loc and if we will call him by his position number “1” we will use df.iloc.
reset_index() #will recreate index column every time we run it on same data
drop = True #paramater won’t create that as column in the dataframe, look at the difference between following two dataset
`inplace = True` save us from assigning it to data again
#we are not using `drop = True`, now df should have its last index as a column in it

# Filter
filtered_df = df[df.column_name == value]

# Multindex
#Ce sont des index à plusieurs niveaux, qu'on peut avoir aussi bien sur les lignes que sur les colonnes.
#Attribution d'un multi-index à 2 niveaux, ici aux colonnes :
df = pandas.DataFrame({'A': [0, 1, 2], 'B': [3, 4, 5],
  'C': [6, 7, 8], 'D': [9, 10, 11]})
df.columns = pandas.MultiIndex(levels = [['a', 'b'], ['A', 'B']],
  codes = [[0, 0, 1, 1], [0, 1, 0, 1]])
#donne:
   a     b    
   A  B  A   B
0  0  3  6   9
1  1  4  7  10
2  2  5  8  11

# Pour plus de détails : http://www.python-simple.com/python-pandas/multiindex.php

### Modification de Dataframes
#on ne peut pas renommer individuellement une colonne : df.columns[0] = 'a' ne marche pas ! (non mutable)
#par contre, on peut renommer l'ensemble des colonnes : df.columns = ['a', 'B']
df.rename(columns = {'A': 'a', 'B': 'b'}) : renomme les colonnes A et B en a et b, mais pas les autres s'il y en a d'autres.
df.rename(columns = lambda x: x.lower()) : renommage en donnant une fonction à appliquer.
df.rename(index = {0: 'a', 1: 'b'}, inplace = True) : on peut aussi utiliser des numéros, ici sur les lignes, et ici en modifiant directement le dataframe.
Pour renommer des colonnes en renvoyant le dataframe avec les colonnes renommées : df.set_axis(['A', 'B', 'C'], axis = 1) (on peut aussi utiliser inplace = True, mais autant utiliser directement df.columns = ['A', 'B' 'C'])
df.reindex(columns = ['B', 'C', 'A']) renvoie le dataframe réordonné par colonne.
df[['B', 'C', 'A']] renvoie aussi le dataframe réordonné.


##### avec assign on met le nom de la nouvelle conolonne sans les ""
df['E'] = [0, 10, 100] # ajoute une nouvelle colonne E avec les valeurs associees 
df.assign(E = df['A'] + df['B'], F = 2 * df['A']) # renvoie une copie du dataframe avec deux nouvelles colonnes E et F (sans modifier le dataframe original). 
df2 = df.assign(E = df['A'] + df['B']).assign(F = 2 * df['E']) # equivalent en enchainant les assign
df2.insert(0, 'C', [8, 4, 8]) # insère une colonne à une position donnée.
# Drop 
del df['A'] : permet de détruire la colonne A.
df2.drop(['a', 'c'], inplace = True) # détruit les lignes d'index 'a' et 'c'
df.drop(['A', 'C'], axis = 1, inplace = True) : permet de détruire plusieurs colonnes en même temps.
df.drop(columns = ['A', 'C'], inplace = True) : alternative à l'indication de l'axis.
df.drop(index = ['a', 'c'], inplace = True) : alternative à l'indication de l'axis (destruction de lignes).   
drop_X_train = X_train.select_dtypes(exclude=['object']) #Exclut les colonnes du type 

# Création du DataFrame
df = pd.DataFrame({
    "Nom": ["Alice", "Bob", "Claire", "David", "Eva", "Frank"],
    "Age": [25, 32, 29, 40, 35, 28],
    "Salaire": [50000, 60000, 58000, 62000, 67000, 45000],
    "Ville": ["Paris", "Lyon", "Marseille", "Toulouse", "Paris", "Lyon"],
    "Prime": [5000, 6000, 5800, 6200, 6700, 4500]
})
# Ajouter les nouvelles lignes
df = df.append({"Nom": "Jean", "Age": 41, "Salaire": 50000, "Ville": "Nantes", "Prime": 5000}, ignore_index=True) # On ajoute ligne par ligne
df = df.append({"Nom": "Roger", "Age": 45, "Salaire": 55000, "Ville": "Toulon", "Prime": 5500}, ignore_index=True)

df.astype(numpy.float64) : renvoie un dataframe avec toutes les colonnes converties dans le type indiqué.

# Modification de valeurs 
df['A'][df['A'] < 2] = 0 #mais souvent, Warning indiquant qu'on modifie une copie d'une slice d'un dataframe, donc à éviter.
df['A'] = df['A'].apply(lambda x: 0 if x < 2 else x) # A PREFERER
#ex:
df = pandas.DataFrame({'A': [1, 3, 5], 'B': [7, 6, 2]})
df['C'] = df.apply(lambda x: 0 if x['A'] > x['B'] else x['A'], axis = 1) #en utilisant toutes les valeurs de la ligne : axis = 1 indique que la valeur passée à la fonction est les lignes (sous forme de Series)

# Valeurs non définies
df = pandas.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 20, 30], 'C': [7, 6, 5]}) # est le dataframe utilisé ici
    A   B  C
0   1 NaN  7
1 NaN  20  6
2   3  30  5

df.dropna(how = 'any') ou df.dropna() # renvoie un dataframe avec les lignes contenant au moins une valeur NaN supprimée.
df.dropna(how = 'all') # supprime les lignes où toutes les valeurs sont NaN.
df.dropna(axis = 1, how = 'any') # supprime les colonnes ayant au moins un NaN plutôt que les lignes (le défaut est axis = 0).
df.dropna(inplace = True) #ne renvoie rien, mais fait la modification en place.
df.fillna(0) : renvoie un dataframe avec toutes les valeurs NaN remplacées par 0.
df['A'].fillna(0, inplace = True) # remplace tous les NA de la colonne A par 0, sur place.
df.isnull() # renvoie un dataframe de booléens, avec True dans toutes les cellules non définies.
df = df.replace(numpy.inf, 99) # remplace les valeurs infinies par 99 (on peut utiliser inplace = True)
To select NaN entries you can use pd.isnull() (or its companion pd.notnull())
reviews[pd.isnull(reviews.country)] # provides the dataset where we have NaN values 
reviews.region_2.fillna("Unknown") # to replace na  
a["region_1"] = a["region_1"].fillna("Unknown") # to fill only a specific column

missing_col = sf_permits.isnull().sum(axis = 0) # Count number of missing values (for each column, by default )
missing_col = sf_permits.isnull().sum(axis = 1) # same for each row
# 
# replace all NA's the value that comes directly after it in the same column, then replace all the remaining na's with 0
subset_nfl_data.fillna(method='bfill', axis=0).fillna(0)

# Valeurs redondantes :
df = pandas.DataFrame({'A': [4, 2, 4, 2], 'B': [7, 3, 7, 3], 'C': [1, 8, 1, 9]})
df.drop_duplicates() # renvoie un dataframe avec les lignes redondantes enlevées en n'en conservant qu'une seule (ici 3 lignes restant)
df.drop_duplicates(keep = False) # renvoie un dataframe avec les lignes redondantes toutes enlevées (ici 2 lignes restant)
df.drop_duplicates(inplace = True) # fait la modification en place.
df.drop_duplicates(subset = ['A', 'B']) # renvoie un dataframe avec les doublons enlevés en considérant seulement les colonnes A et B, et en renvoyant la 1ère ligne pour chaque groupe ayant mêmes valeurs de A et B.
df.drop_duplicates(subset = ['A', 'B'], keep = 'last') # on conserve la dernière ligne plutôt que la première (keep = first, qui est le défaut).

# Tri de dataframe :
df.sort_values(by = 'C') # renvoie un dataframe avec les lignes triées de telle sorte que les valeurs de la colonne 'C' soit dans l'ordre croissant :
df.sort_values(by = ['C', 'A']) # tri selon 2 colonnes successives, d'abord C, puis A.
df.sort_values(by = 'C', ascending = False) : tri par ordre décroissant.
df.sort_values(by = ['C', 'A'], ascending = [True, False]) : tri selon 2 clefs, la première par ordre croissant et la 2ème par ordre décroissant.
df.sort_values(by = 'C', inplace = True) : ne renvoie rien et fait le tri en place.


# Jointures et concaténations
# Concat
pour concaténer 2 dataframes (ou plus) ayant les mêmes colonnes les uns en dessous des autres :
df1 = pandas.DataFrame({'A': [3, 5], 'B': [1, 2]}, index = [0, 1])
df2 = pandas.DataFrame({'A': [6, 7], 'B': [4, 9]}, index = [2, 3])
pandas.concat([df1, df2]) # fonctionne meme si les index sont identiques
#si les dataframes n'ont pas les mêmes colonnes, par défaut, des NaN sont mis aux endroits non définis

pandas.concat([df1, df2], join = 'inner') donne #si les dataframes n'ont pas les mêmes colonnes et qu'on veut conserver seulement les colonnes communes, intersection (sans avoir de NaN) 
df1 = pandas.DataFrame({'A': [3, 5], 'B': [1, 2]}); df2 = pandas.DataFrame({'C': [6, 7], 'D': [4, 9]}); pandas.concat([df1, df2], axis = 1) # juxtaposition de colonnes plutôt que de lignes
pandas.concat([df, pandas.DataFrame(df.sum(), columns = ['total']).T]) #Pour rajouter le total par colonne

# Joins
df1 = pandas.DataFrame({'A': [3, 5], 'B': [1, 2]}); df2 = pandas.DataFrame({'A': [5, 3, 7], 'C': [9, 2, 0]}); 
pandas.merge(df1, df2) #jointure simple (inner) qui par défaut utilise les noms des colonnes qui sont communs : df1.merge(df2) marche aussi
   A  B  C
0  3  1  2
1  5  2  9

#on peut indiquer explicitement les colonnes sur lequelles on veut faire la jointure si c'est une partie seulement des colonnes de même non : 
df1 = pandas.DataFrame({'A': [3, 5], 'B': [1, 2]}); df2 = pandas.DataFrame({'A': [5, 3, 7], 'B': [9, 2, 0]}); 
pandas.merge(df1, df2, on = ['A']) (ou df1.merge(df2, on = ['A'])) donne :
   A  B_x  B_y
0  3    1    2
1  5    2    9
    
###Pivotage de Dataframes
#il faut avoir une seule valeur par combinaison de ligne et de colonne.
#utile notamment si les valeurs sont du texte.
df = pandas.DataFrame({'A': ['a', 'a', 'b', 'b', 'c', 'c'], 'T': ['yes', 'no', 'yes', 'no', 'yes', 'no'], 'V': [4, 2, 5, 2, 7, 3]}) #c'est à dire :
   A    T  V
0  a  yes  4
1  a   no  2
2  b  yes  5
3  b   no  2
4  c  yes  7
5  c   no  3

df.pivot(index = 'A', columns = 'T', values = 'V') #renvoie :
T  no  yes
A         
a   2    4
b   2    5
c   3    7
#si il y a des combinaisons qui manquent, on a un NaN à ces places-là.
#si pour certaines combinaisons on a plusieurs lignes (plusieurs valeurs) : erreur (ValueError)

#pivot_table : plus souple que pivot 
#si pour certaines combinaisons on a plusieurs lignes (plusieurs valeurs), fait la moyenne : 
df = pandas.DataFrame({'A': ['a', 'a', 'b', 'b', 'c', 'c', 'c'], 'T': ['yes', 'no', 'yes', 'no', 'yes', 'no', 'no'], 'V': [4, 2, 5, 2, 7, 3, 1]}); 
df.pivot_table(index = 'A', columns = 'T', values = 'V') donne :
T  no  yes
A         
a   2    4
b   2    5
c   2    7
#on peut utiliser une autre fonction que la moyenne : df.pivot_table(index = 'A', columns = 'T', values = 'V', aggfunc = 'median'). Pour aggfunc, on peut utiliser les noms des fonctions ('min', 'median', ...) ou les fonctions elles-mêmes (min, numpy.median, len).
#Si df est un dataframe avec 2 colonnes 'A' et 'B' et on veut compter le nombre d'ocurrence de chaque paires, avec A en ligne et B en colonne : df.pivot_table(index = 'A', columns = 'B', aggfunc = len).fillna(0)
df.pivot_table(index = 'A', columns = 'T', values = 'V', fill_value = 0) : fill_value indique la valeur à mettre au lieu de NaN pour les combinaisons manquantes.
df.pivot_table(index = 'A', columns = ['B', 'C'], values = 'V') : on peut utiliser plusieurs colonnes.
df.pivot_table(index = 'A', columns = 'B', values = ['V1', 'V2'] : on peut utiliser plusieurs valeurs (on aura alors df.columns qui sera à plusieurs niveaux)

### Calculs d'aggrégats
 df.mean(axis = 0)#  (c'est le défaut) par colonne (moyenne des valeurs de chaque ligne pour une colonne) :
 df.mean(axis = 1)# de toutes les colonnes (une valeur par ligne) :
df.mean(skipna = True)# (si False, on aura NaN à chaque fois qu'il y a au moins une valeur non définie).
df['A'].mean() #  calculer la moyenne pour une seule colonne :
df.std() # écart-type (au sens statistique, avec n - 1, contrairement à numpy.std dont le défaut est l'écart-type avec n : ddof = 0). Saute les valeurs à NaN par défaut (donc renvoie NaN si seulement 0 ou 1 valeur non NaN)
df.groupby("Produit")["Revenu"].sum().idxmax() # getr the prduit insteadf of the value of it
df.select_dtypes(include="number").aggregate(["mean", "sum", "min", "max"]) #include only numbers and provide some basic stats

#Pour standardiser un dataframe :
par colonne : (df - df.mean()) / df.std(ddof = 0)
par ligne : (df - df.apply(numpy.mean, axis = 1, result_type = 'broadcast')) / df.apply(numpy.std, axis = 1, result_type = 'broadcast') ou ((df.T - df.T.mean()) / df.T.std(ddof = 0)).T
# attention : si les valeurs sont entières, faire d'abord df = df.astype(float)

# Group
#groupby() created a group of reviews which allotted the same point values to the given wines. Then, for each of these groups, we grabbed the points() column and counted how many 
#times it appeared. value_counts() is just a shortcut to this groupby() operation.
#You can think of each group we generate as being a slice of our DataFrame containing only data with values that match, so we can use apply quite easily
#On peut grouper un dataframe par une ou plusieurs colonne. Si df = pandas.DataFrame({'A': ['a', 'b', 'a', 'a', 'b'], 'B': [8, 4, 5, 10, 8], 'C': ['x', 'x', 'y', 'y', 'x'], 'D': [0, 1, 2, 3, 4]}) :
df.groupby('A') #: renvoie un objet de la classe pandas.core.groupby.DataFrameGroupBy.
len(df.groupby('A')) : nombre de groupes
df.groupby('A').groups : dictionnaire valeur vers liste des index pour cette valeur.
df.groupby('A').sum() #: groupe avec les valeurs de A et fait la somme, pour les colonnes pour lesquelles c'est possible :
df[["genre","salaire"]].groupby("genre").mean() # pour calculer le salaire moyen par sexe
result = reviews.groupby('winery').apply(lambda df: df.loc[df.price.isna() | (df.price == df.price.max())]) # exemple de vin
#In all of the examples we've seen thus far we've been working with DataFrame or Series objects with a single-label index. groupby() is 
#slightly different in the fact that, depending on the operation we run, it will sometimes result in what is called a multi-index.
#However, in general the multi-index method you will use most often is the one for converting back to a regular index, the reset_index()

#https://moncoachdata.com/blog/groupby-de-pandas/   pour plus d'info


####### Summary Functions and Maps ########
#Summary functions
Pandas provides many simple "summary functions" (not an official name) which restructure the data in some useful way. For example, consider the describe() method:
To see a list of unique values we can use the unique() function
To see a list of unique values and how often they occur in the dataset, we can use the value_counts() method
reviews.taster_name.value_counts() ## Very useful

#Map
review_points_mean = reviews.points.mean()
reviews.points.map(lambda p: p - review_points_mean)
The function you pass to map() should expect a single value from the Series (a point value, in the above example), and return a transformed version of that value. map() returns a new Series where 
all the values have been transformed by your function. apply() is the equivalent method if we want to transform a whole DataFrame by calling a custom method on each row.

def remean_points(row):
    row.points = row.points - review_points_mean
    return row

reviews.apply(remean_points, axis='columns')#If we had called reviews.apply() with axis='index', then instead of passing a function to transform each row, we would need to give a function to transform each column.


####exemple tres intéressant ~#######
reviews["stars"] = reviews.points.map(lambda x:  3 if x >= 95 else 2 if x >= 85 else 1)
reviews.loc[reviews.country == 'Canada', "stars"] = 3
star_ratings = reviews["stars"]
# on aurait pu faire également avec une fonction et apply:
def stars(row):
    if row.country == 'Canada':
        return 3
    elif row.points >= 95:
        return 3
    elif row.points >= 85:
        return 2
    else:
        return 1

star_ratings = reviews.apply(stars, axis='columns')



                        ######## Data Visualization (with Seaborn) ###############   ===> Check seaborn Cheat Sheet

### Set up 
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

Pour le val_accuracy, tu as utilisé y = val_accuracy, x = range(len(val_accuracy)), mais la syntaxe correcte est range(len(val_accuracy)), val_accuracy, 
car x et y sont des arguments positionnels dans plot(), et non des arguments nommés.

plt.subplot(nrows, ncols, index)


#Line plots
    # Set the width and height of the figure
    plt.figure(figsize=(16,6))
    # Add title
    plt.title("Daily Global Streams of Popular Songs in 2017-2018")
    # Line chart showing how songs grow with time
    sns.lineplot(data=spotify_data)

# plot only two songs:
    # Set the width and height of the figure
    plt.figure(figsize=(14,6))
    # Add title
    plt.title("Daily Global Streams of Popular Songs in 2017-2018")
    #The next two lines each add a line to the line chart. For instance, consider the first one, which adds the line
    # Line chart showing daily global streams of 'Shape of You'
    sns.lineplot(data=spotify_data['Shape of You'], label="Shape of You")
    # Line chart showing daily global streams of 'Despacito'
    sns.lineplot(data=spotify_data['Despacito'], label="Despacito")
    # Add label for horizontal axis
    plt.xlabel("Date")

# Bar Charts
# Set the width and height of the figure
plt.figure(figsize=(10,6))
# Add title
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")
# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=flight_data.index, y=flight_data['NK'])
# Add label for vertical axis
plt.ylabel("Arrival delay (in minutes)")

# Haetmap
# Set the width and height of the figure
plt.figure(figsize=(14,7))
# Add title
plt.title("Average Arrival Delay for Each Airline, by Month")
# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=flight_data, annot=True)  # This ensures that the values for each cell appear on the chart. 
# Add label for horizontal axis
plt.xlabel("Airline")

# Scatter plots
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])
sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges']) # Same with a regression line
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker']) # hue means colors, here baased on the Y/N for smoking
sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data) # Now we have the colors plus one regression line for each of these colors
#sns.lmplot command above works slightly differently than the commands you have learned about so far:
#Instead of setting x=insurance_data['bmi'] to select the 'bmi' column in insurance_data, we set x="bmi" to specify the name of the column only.
#Similarly, y="charges" and hue="smoker" also contain the names of columns.
#We specify the dataset with data=insurance_data

# Histograms
sns.histplot(iris_data['Petal Length (cm)'])
sns.histplot(data=iris_data, x='Petal Length (cm)', hue='Species') #adding color coding(hue)

#Density plots a/k/a KDE (a smoothed histogram)
sns.kdeplot(data=iris_data['Petal Length (cm)'], shade=True)
sns.kdeplot(data=iris_data, x='Petal Length (cm)', hue='Species', shade=True)#adding color coding
# 2D KDE plot
sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde")

# subplots
fig, ax = plt.subplots(1, 2, figsize=(15, 3))#Creates a figure (fig) with two subplots (ax[0] and ax[1]) arranged in a single row.
sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(scaled_data, ax=ax[1], kde=True, legend=False)
ax[1].set_title("Scaled data")
plt.show()



##### Useful commands in Seaborn 
plt.show() #Display all open figures

fig = sns.barplot(x=ign_data.index ,y=ign_data.Racing) # for our example
fig.tick_params(axis='x', rotation=90) # allow the figure fig to rotate the x-labels in order to see it clearly
sns.set_style("dark")# Change the style of the figure to the "dark" theme

########## Scaling and Normlization ###########

##SCALING
#This means that you're transforming your data so that it fits within a specific scale, like 0-100 or 0-1. You want to scale data when you're using methods based on measures
# of how far apart data points are, like support vector machines (SVM) or k-nearest neighbors (KNN). With these algorithms, a change of "1" in any numeric feature is given the same importance.
#For example, you might be looking at the prices of some products in both Yen and US Dollars. One US Dollar is worth about 100 Yen, but if you don't scale your 
#prices, methods like SVM or KNN will consider a difference in price of 1 Yen as important as a difference of 1 US Dollar! This clearly doesn't fit with our intuitions of the world. With currency,
# you can convert between currencies. But what about if you're looking at something like height and weight? It's not entirely clear how many pounds should equal one inch 
# (or how many kilograms should equal one meter). By scaling your variables, you can help compare different variables on equal footing.

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns=[0])

##Normalization
#Scaling just changes the range of your data. Normalization is a more radical transformation. The point of normalization is to change your observations so that they can be described as a normal distribution.
#Normal distribution: Also known as the "bell curve", this is a specific statistical distribution where a roughly equal observations fall above and below the mean, the mean and the median are the same,
# and there are more observations closer to the mean. The normal distribution is also known as the Gaussian distribution.
#In general, you'll normalize your data if you're going to be using a machine learning or statistics technique that assumes your data is normally distributed. Some examples of these include linear
# discriminant analysis (LDA) and Gaussian naive Bayes. (Pro tip: any method with "Gaussian" in the name probably assumes normality.) The method we're using to normalize here is called the Box-Cox Transformation.

# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

#example
# get the index of all positive pledges (Box-Cox only takes positive values)
index_of_positive_pledges = kickstarters_2017.usd_pledged_real > 0
# get only positive pledges (using their indexes)
positive_pledges = kickstarters_2017.usd_pledged_real.loc[index_of_positive_pledges]
# normalize the pledges (w/ Box-Cox)
normalized_pledges = pd.Series(stats.boxcox(positive_pledges)[0], 
                               name='usd_pledged_real', index=positive_pledges.index)
print('Original data\nPreview:\n', positive_pledges.head())
print('Minimum value:', float(positive_pledges.min()),
      '\nMaximum value:', float(positive_pledges.max()))
print('_'*30)

print('\nNormalized data\nPreview:\n', normalized_pledges.head())
print('Minimum value:', float(normalized_pledges.min()),
      '\nMaximum value:', float(normalized_pledges.max()))


###### Date Parsing ######

# check the data type of our date column
landslides['date'].dtype
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format="%m/%d/%y")
landslides['date_parsed'] = pd.to_datetime(landslides['Date'], infer_datetime_format=True) # Python can directly infer the date format
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day

#Sometime we have differences in the format of dates in the data entries => goodway to check with len()
date_lengths = earthquakes.Date.str.len() # Take date column, covert it to str in order to count the size
date_lengths.value_counts() # we can see here how many kind of len() we have in the dates

# example of date vizu
shelter_outcomes['date_of_birth'].value_counts().sort_values().plot.line() #plots number of animal birth dates in a shelter (noisy)
# idea: Resample it form by day to by year, to maybe get a clearer plot
shelter_outcomes['date_of_birth'].value_counts().resample('Y').sum().plot.line() # better!

## Lag plot
#One of these plot types is the lag plot. A lag plot compares data points from each observation in the dataset against data points from a previous observation.
#So for example, data from December 21st will be compared with data from December 20th, which will in turn be compared with data from December 19th, and so on. For example, 
#here is what we see when we apply a lag plot to the volume (number of trades conducted) in the stock data:

from pandas.plotting import lag_plot
lag_plot(stocks['volume'].tail(250)) # 
#Time-series data tends to exhibit a behavior called periodicity: rises and peaks in the data that are correlated with time. 
#For example, a gym would likely see an increase in attendance at the end of every workday, hence exhibiting a periodicity of a day. 
#A bar would likely see a bump in sales on Friday, exhibiting periodicity over the course of a week. And so on.

## Autocorrelation plot
#A plot type that takes this concept and goes even further with it is the autocorrelation plot. The autocorrelation plot is a multivariate summarization-type plot
#that lets you check every periodicity at the same time. It does this by computing a summary statistic—the correlation score—across every possible lag in the dataset. 
#This is known as autocorrelation. In an autocorrelation plot the lag is on the x-axis and the autocorrelation score is on the y-axis. 
#The farther away the autocorrelation is from 0, the greater the influence that records that far away from each other exert on one another.
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(stocks['volume'])

####### Character Encoding ######

#Character encoding mismatches are less common today than they used to be, but it's definitely still a problem. There are lots of different character encodings, but the main one you need to know is UTF-8.
#UTF-8 is the standard text encoding. All Python code is in UTF-8 and, ideally, all your data should be as well. It's when things aren't in UTF-8 that you run into trouble.
# start with a string
before = "This is the euro symbol: €"
# check to see what datatype it is
type(before)

#The other data is the bytes data type, which is a sequence of integers. You can convert a string into bytes by specifying which encoding it's in:
# encode it to a different encoding, replacing characters that raise errors
after = before.encode("utf-8", errors="replace")
# check the type
type(after)

# convert it back to utf-8
print(after.decode("utf-8"))

####### Inconsistent Data Entry ##########

# little tips that solves 80% of inconsistencies with string
# convert to lower case
professors['Country'] = professors['Country'].str.lower()
# remove trailing white spaces (at beginning and end of string)
professors['Country'] = professors['Country'].str.strip()

# fuzzywuzzy package to help identify which strings are closest to each other
#Fuzzy matching: The process of automatically finding text strings that are very similar to the target string. In general, a string is considered "closer" to another one the 
#fewer characters you'd need to change if you were transforming one string into another. So "apple" and "snapple" are two changes away from each other (add "s" and "n") while 
#"in" and "on" and one change away (rplace "i" with "o"). You won't always be able to rely on fuzzy matching 100%, but it will usually end up saving you at least a little time.

# get the top 10 closest matches to "south korea"
matches = fuzzywuzzy.process.extract("south korea", countries, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
# take a look at them
matches



                                                       ####################### Machine Learning #####################

##Building Your Model
#You will use the scikit-learn library to create your models. When coding, this library is written as sklearn, as you will see in the sample code.
#Scikit-learn is easily the most popular library for modeling the types of data typically stored in DataFrames.

#The steps to building and using a model are:

#Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
#Fit: Capture patterns from provided data. This is the heart of modeling.
#Predict: Just what it sounds like
#Evaluate: Determine how accurate the model's predictions are.

#target
y = melbourne_data.Price
#features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor
# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)# Randomstate == random seed
# Fit model
melbourne_model.fit(X, y)
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))

## Model Validation¶
#You'll want to evaluate almost every model you ever build. In most (though not all) applications, the relevant measure of model quality is predictive accuracy. 
#In other words, will the model's predictions be close to what actually happens.
# There are many metrics for summarizing model quality, but we'll start with one called Mean Absolute Error (also called MAE).
#error=actual−predicted
from sklearn.metrics import mean_absolute_error
predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices) ## Issue with this: train and evaluate with the same data

#Since models' practical value come from making predictions on new data, we measure performance on data that wasn't used to build the model. 
#The most straightforward way to do this is to exclude some data from the model-building process, and then use those to test the model's accuracy on data it hasn't seen before. 
#This data is called validation data.

    from sklearn.model_selection import train_test_split
    # split data into training and validation data, for both features and target
    # The split is based on a random number generator. Supplying a numeric value to
    # the random_state argument guarantees we get the same split every time we
    # run this script.
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
    # Define model
    melbourne_model = DecisionTreeRegressor()
    # Fit model
    melbourne_model.fit(train_X, train_y)
    # get predicted prices on validation data
    val_predictions = melbourne_model.predict(val_X)
    print(mean_absolute_error(val_y, val_predictions))

# Overfitting and underfitting
#In practice, it's not uncommon for a tree to have 10 splits between the top level (all houses) and a leaf. As the tree gets deeper, the dataset gets sliced up into leaves
# with fewer houses. If a tree only had 1 split, it divides the data into 2 groups. If each group is split again, we would get 4 groups of houses. Splitting each of
# those again would create 8 groups. If we keep doubling the number of groups by adding more splits at each level, we'll have  2¨10 groups of houses by the time we get to the 10th level. That's 1024 leaves.
#When we divide the houses amongst many leaves, we also have fewer houses in each leaf. Leaves with very few houses will make predictions that are quite close to those
# homes' actual values, but they may make very unreliable predictions for new data (because each prediction is based on only a few houses).

# We can use a utility function to help compare MAE scores from different values for max_leaf_nodes:
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

## Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))

### Missing values in a model
#1) A Simple Option: Drop Columns with Missing Values
#2) A Better Option: Imputation (with mean for example)
#3) An Extension To Imputation

# 1) Drop: 
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

# 2) Imputation with mean
#we use SimpleImputer to replace missing values with the mean value along each column
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# 3) Extension to imputation
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()
# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()
# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))
# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

###Categprical Variables =>3 options
# 1) Drop (not a good one)
# 2) Ordinal Encoding
#this one works if it's possible to change categories to numerical ones (like never < often for example) but this doesnt work for colours for example.
# 3) ONE HOT ENCODING (most important one)
#One-hot encoding creates new columns indicating the presence (or absence) of each possible value in the original data. To understand this, we'll work through an example.
# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
Categorical variables:
['Type', 'Method', 'Regionname']
nunique() : compte le nombre de valeuurs uniques (df["A"].nunique())
list(set(object_cols) - set(low_cardinality_cols)) :# set crée un ensemble des elements unique de objects... sans respecter l'ordre. donc on a les elements de objectcols - ceux de low... et on crée une liste de cet ensemble

#We use the OneHotEncoder class from scikit-learn to get one-hot encodings. There are a number of parameters that can be used to customize its behavior.
#We set handle_unknown='ignore' to avoid errors when the validation data contains classes that aren't represented in the training data, and
#setting sparse=False ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).
#To use the encoder, we supply only the categorical columns that we want to be one-hot encoded. For instance, to encode the training data, we supply X_train[object_cols].
# (object_cols in the code cell below is a list of the column names with categorical data, and so X_train[object_cols] contains all of the categorical data in the training set.)

    from sklearn.preprocessing import OneHotEncoder
    # Apply one-hot encoder to each column with categorical data
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols])) #Utilisez fit_transform sur les données d'entraînement.
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols])) # Utilisez transform sur les données de validation ou de test.

    # One-hot encoding removed index; put it back
    OH_cols_train.index = X_train.index
    OH_cols_valid.index = X_valid.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_X_train = X_train.drop(object_cols, axis=1)
    num_X_valid = X_valid.drop(object_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

    # Ensure all columns have string type
    OH_X_train.columns = OH_X_train.columns.astype(str)
    OH_X_valid.columns = OH_X_valid.columns.astype(str)

    print("MAE from Approach 3 (One-Hot Encoding):") 
    print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))


# différence avec get dummies et factorize:
import pandas as pd

df = pd.DataFrame({'category': ['A', 'B', 'A', 'C']})
df_dummies = pd.get_dummies(df, columns=['category'], drop_first=True)
print(df_dummies)

# Label encoding for categoricals with factorize
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

#Avec OneHotEncoder :
from sklearn.preprocessing import OneHotEncoder
import numpy as np

encoder = OneHotEncoder(drop='first')
X = np.array([['A'], ['B'], ['A'], ['C']])
X_encoded = encoder.fit_transform(X).toarray()
print(X_encoded)
#En résumé, get_dummies est pratique pour des transformations simples sur des DataFrames,
# tandis que OneHotEncoder est plus adapté pour une utilisation dans des pipelines de machine learning, offrant plus de flexibilité et de contrôle sur le prétraitement des données.
# Factorize est simple à utiliser mais crée des relations implicites (vert=0, bleu=1 => ordinalité) qui pourraient preter à confusiion dans certains algo.

### Pipelines
#Pipelines are a simple way to keep your data preprocessing and modeling code organized. Specifically, a pipeline bundles preprocessing and modeling steps so you can use
# the whole bundle as if it were a single step.
#Many data scientists hack together models without pipelines, but pipelines have some important benefits. Those include:
#Cleaner Code: Accounting for data at each step of preprocessing can get messy. With a pipeline, you won't need to manually keep track of your training and validation data at each step.
#Fewer Bugs: There are fewer opportunities to misapply a step or forget a preprocessing step.
#Easier to Productionize: It can be surprisingly hard to transition a model from a prototype to something deployable at scale. We won't go into the many related concerns here, 
#but pipelines can help.
#More Options for Model Validation: You will see an example in the next tutorial, which covers cross-validation.

##Step 1: Define Preprocessing Steps¶
#Similar to how a pipeline bundles together preprocessing and modeling steps, we use the ColumnTransformer class to bundle together different preprocessing steps. The code below:
imputes missing values in numerical data, and
imputes missing values and applies a one-hot encoding to categorical data.

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='constant')

    # Preprocessing for categorical data (impute missing and then one hot encoder)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

##Step 2: Define the Model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=0)

##Step 3: Create and Evaluate the Pipeline
#Finally, we use the Pipeline class to define a pipeline that bundles the preprocessing and modeling steps. There are a few important things to notice:
#With the pipeline, we preprocess the training data and fit the model in a single line of code. (In contrast, without a pipeline, we have to do imputation,
# one-hot encoding, and model training in separate steps. This becomes especially messy if we have to deal with both numerical and categorical variables!)
#With the pipeline, we supply the unprocessed features in X_valid to the predict() command, and the pipeline automatically preprocesses the features before generating predictions.
#(However, without a pipeline, we have to remember to preprocess the validation data before making predictions.)

    from sklearn.metrics import mean_absolute_error

    # Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)
                                 ])
    # Preprocessing of training data, fit model 
    my_pipeline.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    preds = my_pipeline.predict(X_valid)

    # Evaluate the model
    score = mean_absolute_error(y_valid, preds)
    print('MAE:', score)



### Cross Validation

#When should you use cross-validation?¶
#Cross-validation gives a more accurate measure of model quality, which is especially important if you are making a lot of modeling decisions. However, it can take longer to run, because it estimates multiple models (one for each fold).
#So, given these tradeoffs, when should you use each approach?
#For small datasets, where extra computational burden isn't a big deal, you should run cross-validation.
#For larger datasets, a single validation set is sufficient. Your code will run faster, and you may have enough data that there's little need to re-use some of it for holdout.
#There's no simple threshold for what constitutes a large vs. small dataset. But if your model takes a couple minutes or less to run, it's probably worth switching to cross-validation.
#Alternatively, you can run cross-validation and see if the scores for each experiment seem close. If each experiment yields the same results, a single validation set is probably sufficient.

#While it's possible to do cross-validation without pipelines, it is quite difficult! Using a pipeline will make the code remarkably straightforward.
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                                  ('model', RandomForestRegressor(n_estimators=50,
                                                                  random_state=0))
                                 ])
    # cross_val_score() from sickit learn
    from sklearn.model_selection import cross_val_score

    # Multiply by -1 since sklearn calculates *negative* MAE
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=5,
                                  scoring='neg_mean_absolute_error') # On met le pipeline direct à manger dans le CV/

    print("MAE scores:\n", scores)

### XGboost
#We refer to the random forest method as an "ensemble method". By definition, ensemble methods combine the predictions of several models 
#(e.g., several trees, in the case of random forests).
#Next, we'll learn about another ensemble method called gradient boosting.
#
#Gradient boosting is a method that goes through cycles to iteratively add models into an ensemble.

#It begins by initializing the ensemble with a single model, whose predictions can be pretty naive. (Even if its predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors.)

#Then, we start the cycle:

#First, we use the current ensemble to generate predictions for each observation in the dataset. To make a prediction, we add the predictions from all models in the ensemble.
#These predictions are used to calculate a loss function (like mean squared error, for instance).
#Then, we use the loss function to fit a new model that will be added to the ensemble. Specifically, we determine model parameters so that adding this new model to the ensemble 
#will reduce the loss. (Side note: The "gradient" in "gradient boosting" refers to the fact that we'll use gradient descent on the loss function to determine the parameters in this new model.)
#Finally, we add the new model to ensemble, and ...   ... repeat!

from xgboost import XGBRegressor

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and  X_train_full[cname].dtype == "object"]
# interesting command for keeping only low card categorical variables.
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']] # to select num var 
 

my_model = XGBRegressor()
my_model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

# Parameter Tuning
#n_estimators specifies how many times to go through the modeling cycle described above. It is equal to the number of models that we include in the ensemble.

#Too low a value causes underfitting, which leads to inaccurate predictions on both training data and test data.
#Too high a value causes overfitting, which causes accurate predictions on training data, but inaccurate predictions on test data (which is what we care about).
#early_stopping_rounds offers a way to automatically find the ideal value for n_estimators. Early stopping causes the model to stop iterating when the validation score stops improving
#Setting early_stopping_rounds=5 is a reasonable choice. In this case, we stop after 5 straight rounds of deteriorating validation scores.

#learning_rate¶
#Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number 
#(known as the learning rate) before adding them in.
#In general, a small learning rate and large number of estimators will yield more accurate XGBoost models, though it will also take the model longer to train 
#since it does more iterations through the cycle. As default, XGBoost sets learning_rate=0.1

    my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs = 4) # n_jobs works for how many cores of your cpu are working on it (parallelism)
    my_model.fit(X_train, y_train, 
                 early_stopping_rounds=5, 
                 eval_set=[(X_valid, y_valid)], 
                 verbose=False)




                                                       ####################### Model Interpretations #####################
# Why is it so important to get information regarding which features are most important in a prediction for example?
Debugging
Informing feature engineering
Directing future data collection
Informing human decision-making
Building Trust

### Feature importance:

## 1) Permutation importance: l'idée est de shuffler une colonne et voir l'impact sur la loss-fction du modele; ce qui en définitive nous donne l'importance de la variable.
#Get a trained model.
#Shuffle the values in a single column, make predictions using the resulting dataset. Use these predictions and the true target values to calculate how much the loss function suffered from shuffling.
# That performance deterioration measures the importance of the variable you just shuffled.
#Return the data to the original order (undoing the shuffle from step 2). Now repeat step 2 with the next column in the dataset, until you have calculated the importance of each column.
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    my_model = RandomForestClassifier(n_estimators=100,
                                      random_state=0).fit(train_X, train_y)
    import eli5
    from eli5.sklearn import PermutationImportance
    perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y) # on permute sur les données de validation
    eli5.show_weights(perm, feature_names = val_X.columns.tolist())

## 2) Partial Plots

#While feature importance shows what variables most affect predictions, partial dependence plots show how a feature affects predictions.
#This is useful to answer questions like:
#Controlling for all other house features, what impact do longitude and latitude have on home prices? To restate this, how would similarly sized houses be priced in different areas?
#Are predicted health differences between two groups due to differences in their diets, or due to some other factor?
    
    from matplotlib import pyplot as plt
    from sklearn.inspection import PartialDependenceDisplay
    # Create and plot the data
    disp1 = PartialDependenceDisplay.from_estimator(tree_model, val_X, ['Goal Scored'])
    plt.show()

    # Code to plot all PDPs at once
    for feat_name in base_features:
    PartialDependenceDisplay.from_estimator(first_model, val_X, [feat_name])
    plt.show()

# 2-D Partial Dependence Plot ==> Allow to see interactions between 2 features.
    fig, ax = plt.subplots(figsize=(8, 6))
    f_names = [('Goal Scored', 'Distance Covered (Kms)')]
    # Similar to previous PDP plot except we use tuple of features instead of single feature
    disp4 = PartialDependenceDisplay.from_estimator(tree_model, val_X, f_names, ax=ax)
    plt.show()


### 3) SHAP Values
# !!!!!!!!!!!!!!!! When plotting, we call shap_values[1]. For classification problems, there is a separate array of SHAP values for each possible outcome.
# In this case, we index in to get the SHAP values for the prediction of "True". !!!!!!!!!!!!!!!!!!!!

#You've seen (and used) techniques to extract general insights from a machine learning model. But what if you want to break down how the model works for an individual prediction?
#EX:
#A model says a bank shouldn't loan someone money, and the bank is legally required to explain the basis for each loan rejection
#A healthcare provider wants to identify what factors are driving each patient's risk of some disease so they can directly address those risk factors with targeted health interventions
#SHAP values interpret the impact of having a certain value for a given feature in comparison to the prediction we'd make if that feature took some baseline value.

#How much was a prediction driven by the fact that the team scored 3 goals, instead of some baseline number of goals.
#Of course, each team has many features. So if we answer this question for number of goals, we could repeat the process for all other features.
#SHAP values do this in a way that guarantees a nice property. Specifically, you decompose a prediction with the following equation:
    sum(SHAP values for all features) = pred_for_team - pred_for_baseline_values #for the soccer example
#If you subtract the length of the blue bars from the length of the pink bars, it equals the distance from the base value to the output.

    my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
    row_to_show = 5 #arbitraire
    data_for_prediction = val_X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
    my_model.predict_proba(data_for_prediction_array) #==> array([[0.29, 0.71]]), The team is 70% likely to have a player win the award.
    #Now, we'll move onto the code to get SHAP values for that single prediction
    import shap  # package used to calculate Shap values
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(my_model)
    # Calculate Shap values
    shap_values = explainer.shap_values(data_for_prediction)
    #The shap_values object above is a list with two arrays. The first array is the SHAP values for a negative outcome (don't win the award), and the second array 
    #is the list of SHAP values for the positive outcome (wins the award). We typically think about predictions in terms of the prediction of a positive outcome, 
    #so we'll pull out SHAP values for positive outcomes (pulling out shap_values[1]).
    # Visualize
    shap.initjs()
    shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)

#shap.TreeExplainer(my_model). But the SHAP package has explainers for every type of model like:
shap.DeepExplainer works with Deep Learning models.
shap.KernelExplainer works with all models, though it is slower than other Explainers and it offers an approximation rather than exact Shap values.

## SummaryPlots
#Permutation importance is great because it created simple numeric measures to see which features mattered to a model. This helped us make comparisons between features easily, 
#and you can present the resulting graphs to non-technical audiences.
#But it doesn't tell you how each features matter. If a feature has medium permutation importance, that could mean it has:
#a large effect for a few predictions, but no effect in general, or a medium effect for all predictions.

#SHAP summary plots give us a birds-eye view of feature importance and what is driving it. We'll walk through an example plot for the soccer data:
This plot is made of many dots. Each dot has three characteristics:
-Vertical location shows what feature it is depicting
-Color shows whether that feature was high or low for that row of the dataset
-Horizontal location shows whether the effect of that value caused a higher or lower prediction.

    import shap  # package used to calculate Shap values
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(my_model)
    # calculate shap values. This is what we will plot.
    # Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
    shap_values = explainer.shap_values(val_X)
    # Make plot. Index of [1] is explained in text below.
    shap.summary_plot(shap_values[1], val_X) 

## SHAP Dependence Contribution Plots
#We've previously used Partial Dependence Plots to show how a single feature impacts predictions. These are insightful and relevant for many real-world use cases.
# Plus, with a little effort, they can be explained to a non-technical audience.
#But there's a lot they don't show. For instance, what is the distribution of effects? Is the effect of having a certain value pretty constant, 
#or does it vary a lot depending on the values of other feaures. SHAP dependence contribution plots provide a similar insight to PDP's, but they add a lot more detail.
    import shap  # package used to calculate Shap values
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(my_model)
    # calculate shap values. This is what we will plot.
    shap_values = explainer.shap_values(X)
    # make plot.
    shap.dependence_plot('Ball Possession %', shap_values[1], X, interaction_index="Goal Scored")


QUESTION:# est-il fiable d'utiliser ces méthodes pour un modele de moyenne ou basse accuracy??



                                                       #######################Feature Engineering #####################

You might perform feature engineering to:
#-improve a model's predictive performance
#-reduce computational or data needs
#-improve interpretability of the results

Example:
# When there are non linear relationships (but quadratic ones for example) go from length to area to predict prices.

# The metric we'll use is called "mutual information". Mutual information is a lot like correlation in that it measures a relationship between two quantities.
# The advantage of mutual information is that it can detect any kind of relationship, while correlation only detects linear relationships.
#MI can help you to understand the relative potential of a feature as a predictor of the target, considered by itself.
#It's possible for a feature to be very informative when interacting with other features, but not so informative all alone. 
#MI can't detect interactions between features. It is a univariate metric.

    from sklearn.feature_selection import mutual_info_regression

    def make_mi_scores(X, y, discrete_features):
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores

    mi_scores = make_mi_scores(X, y, discrete_features)
    mi_scores[::3]  # show a few features with their MI scores (1 over 3)

# Plotting with barplot
    def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")

    plt.figure(dpi=100, figsize=(8, 5))
    plot_mi_scores(mi_scores)

# After getting the information about the interesting variables, plot is quite often a good option:
sns.relplot(x="curb_weight", y="price", data=df);

#Note
#Before deciding a feature is unimportant from its MI score, it's good to investigate any possible interaction effects

### Creating Features
#Tips on Discovering New Features:
#Understand the features. Refer to your dataset's data documentation, if available.
#Research the problem domain to acquire domain knowledge. If your problem is predicting house prices, do some research on real-estate for instance.
#Study previous work. Solution write-ups from past Kaggle competitions are a great resource.
#Use data visualization. Visualization can reveal pathologies in the distribution of a feature or complicated relationships that could be simplified.

# Mathematical Transfo 
autos["stroke_ratio"] = autos.stroke / autos.bore #gives info regarding the efficiency vs performance.
#The more complicated a combination is, the more difficult it will be for a model to learn, like this formula for an engine's "displacement", a measure of its power
autos["displacement"] = (
    np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos.num_of_cylinders) 

Data visualization can suggest transformations, often a "reshaping" of a feature through powers or logarithms (effective at normalizing it)
# If the feature has 0.0 values, use np.log1p (log(1+x)) instead of np.log
    accidents["LogWindSpeed"] = accidents.WindSpeed.apply(np.log1p)
    # Plot a comparison
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    sns.kdeplot(accidents.WindSpeed, shade=True, ax=axs[0])
    sns.kdeplot(accidents.LogWindSpeed, shade=True, ax=axs[1]);

#In Traffic Accidents are several features indicating whether some roadway object was near the accident.
# This will create a count of the total number of roadway features nearby using the sum method:
    roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay",
        "Junction", "NoExit", "Railway", "Roundabout", "Station", "Stop",
        "TrafficCalming", "TrafficSignal"]
    accidents["RoadwayFeatures"] = accidents[roadway_features].sum(axis=1)

    accidents[roadway_features + ["RoadwayFeatures"]].head(10)

#Many formulations lack one or more components (the component has a value of 0). This will count how many components are in a formulation with the dataframe's built-in gt method
    components = [ "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
                   "Superplasticizer", "CoarseAggregate", "FineAggregate"]
    concrete["Components"] = concrete[components].gt(0).sum(axis=1)

    concrete[components + ["Components"]].head(10)

#The str accessor lets you apply string methods like split directly to columns.
    customer[["Type", "Level"]] = (  # Create two new features
        customer["Policy"]           # from the Policy feature
        .str                         # through the string accessor
        .split(" ", expand=True)     # by splitting on " "
                                     # and expanding the result into separate columns
    )
    customer[["Policy", "Type", "Level"]].head(10)

# Opération inverse
    autos["make_and_style"] = autos["make"] + "_" + autos["body_style"]

# Or create feature by groups use the groupby and transform
    customer["AverageIncome"] = (
        customer.groupby("State")  # for each state
        ["Income"]                 # select the income
        .transform("mean")         # and compute its mean
    )

#Here's how you could calculate the frequency with which each state occurs in the dataset:
    customer["StateFreq"] = (customer.groupby("State")["State"].transform("count")/customer.State.count())

    X_5["MedNhbdArea"] =X.groupby("Neighborhood").GrLivArea.median() # Ici on a en index des lignes, les Neigboorhood unique (28lignes), avec en valeur de colonne la median de GRLIVarea
    X_5["MedNhbdArea"] =X.groupby("Neighborhood").GrLivArea.transform("median") # Alors que là on a cette meme médiane mais pour chaque entrée réelle du dataset (2930 lignes), avec répétition de la valeurs pour les meme neigboorhood


#Tips on Creating Features
-#It's good to keep in mind your model's own strengths and weaknesses when creating features. Here are some guidelines:
-#Linear models learn sums and differences naturally, but can't learn anything more complex.
-#Ratios seem to be difficult for most models to learn. Ratio combinations often lead to some easy performance gains.
-#Linear models and neural nets generally do better with normalized features. Neural nets especially need features scaled to values not too far from 0.
-#Tree-based models (like random forests and XGBoost) can sometimes benefit from normalization, but usually much less so.
-#Tree models can learn to approximate almost any combination of features, but when a combination is especially important they can still benefit from having it explicitly created, especially when data is limited.
-#Counts are especially helpful for tree models, since these models don't have a natural way of aggregating information across many features at once.


### Clustering with K-means

- #Unsupervised learning (no target), => feature discovery

#It's a simple two-step process. The algorithm starts by randomly initializing some predefined number (n_clusters) of centroids. It then iterates over these two operations:
1-#assign points to the nearest cluster centroid
2-#move each centroid to minimize the distance to its points
#It iterates over these two steps until the centroids aren't moving anymore, or until some maximum number of iterations has passed (max_iter).

kmeans = KMeans(n_clusters=6)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")
sns.relplot(    x="Longitude", y="Latitude", hue="Cluster", data=X, height=6,) # plot
X["MedHouseVal"] = df["MedHouseVal"]
sns.catplot(x="MedHouseVal", y="Cluster", data=X, kind="boxen", height=6) #MedHouse = target
# If the clustering is informative, these distributions should, for the most part, separate across MedHouseVal, which is indeed what we see.



### Principal Component Analysis
- #Unsupervised as well
- #PCA is typically applied to standardized data. With standardized data "variation" means "correlation". With unstandardized data "variation" means "covariance". 
#All data in this course will be standardized before applying PCA.Removing outliers could be important as well. 
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

-The whole idea of PCA: instead of describing the data with the original features, we describe it with its axes of variation. The axes of variation become the new features.
df["Size"] = 0.707 * X["Height"] + 0.707 * X["Diameter"] # PC1
df["Shape"] = 0.707 * X["Height"] - 0.707 * X["Diameter"] # PC2
# Can have as many PCs as original features, but explained variance goes decreasing for each new PCs.
ExpVar pC1 = 90%
ExpaVar Pc2 = 6%
...

There are two ways you could use PCA for feature engineering:

1- use it as a descriptive technique.
2- use the components themselves as features, they can often be more informative than the original features.
=> use cases: 
-Dimensionality reduction:When your features are highly redundant (multicollinear, specifically), PCA will partition out the redundancy into one or more near-zero variance components, which you can then drop since they will contain little or no information.
-Anomaly detection:Unusual variation, not apparent from the original features, will often show up in the low-variance components
-Noise reduction
-Decorrelation: PCA transforms correlated features into uncorrelated components, which could be easier for your algorithm to work with.

from sklearn.decomposition import PCA
# Create principal components
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
# Convert to dataframe
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])] #using f-string
X_pca = pd.DataFrame(X_pca, columns=component_names)

# Then We'll wrap the loadings up in a dataframe (loading = weights)
loadings = pd.DataFrame(
    pca.components_.T,  # transpose the matrix of loadings
    columns=component_names,  # so the columns are the principal components
    index=X.columns,  # and the rows are the original features
)
loadings
# Look at explained variance
plot_variance(pca);
# Look at Mutual Info for these new features 
mi_scores = make_mi_scores(X_pca, y, discrete_features=False)
mi_scores


###Target Encoding
#A target encoding is any kind of encoding that replaces a feature's categories with some number derived from the target.
https://www.kaggle.com/code/ryanholbrook/target-encoding

#L'idée principale du target encoding est de remplacer les catégories d'une variable catégorielle par une statistique (par exemple, la moyenne ou médiane) 
#calculée à partir de la variable cible.

Exemple :
#Si vous avez une variable catégorielle Ville et une variable cible Prix, au lieu d'encoder Ville avec un encodage classique (comme one-hot encoding), 
#vous pourriez calculer la moyenne des prix pour chaque ville et remplacer chaque ville par cette moyenne.
En utilisant le target encoding, vous remplacez chaque ville par la moyenne des prix associés à cette ville.
Dans cet exemple, Paris devient 150, Lyon devient 200, et Marseille devient 300.

Avantages:
-#Gère les variables catégorielles à haute cardinalité : Contrairement à l'encodage one-hot (qui crée une colonne pour chaque catégorie), le target encoding ne crée 
#qu'une seule colonne numérique, ce qui est particulièrement utile lorsque la variable catégorielle contient un grand nombre de catégories.
-#Capturer la relation avec la cible : Contrairement à l'encodage traditionnel (où chaque catégorie est traitée de manière égale), le target encoding exploite 
#la relation entre chaque catégorie et la variable cible, ce qui peut améliorer les performances du modèle.

Inconvénients:
-#Overfitting (surapprentissage) : Comme cette méthode utilise des informations provenant de la cible, il y a un risque de surapprentissage, 
#notamment si vous encodez les variables sur le même ensemble de données que celui utilisé pour l'entraînement du modèle. Par exemple, le modèle pourrait mémoriser des 
#moyennes spécifiques qui ne se généralisent pas bien à de nouvelles données.

-#Lenteur avec de nombreuses catégories : Si vous avez beaucoup de catégories et peu d'exemples pour certaines d'entre elles, la moyenne ou la statistique pourrait 
#ne pas être fiable. Il est donc nécessaire de régulariser l'estimation pour éviter un biais trop important sur les petites catégories.

# Jeu de données exemple
data = pd.DataFrame({
    'Ville': ['Paris', 'Paris', 'Lyon', 'Lyon', 'Marseille'],
    'Prix': [100, 200, 150, 250, 300]
})

# Target encoding
encoder = ce.TargetEncoder(cols=['Ville'])
data['Ville_encoded'] = encoder.fit_transform(data['Ville'], data['Prix'])

print(data)


 #F-strings                                                       ####################### Generic Usefull Comands #####################
print(f"Le résultat de 5 + 10 est égal à {5 + 10}")
prenom = "Patrick"
print(f"Bonjour {prenom}")

#PolynomialFeatures Sklearn
#Utilisation d'une régression polynomiale (via )
#Une autre approche consiste à utiliser les polynomial features de la bibliothèque sklearn


#################### Creation automtique de variables globales vua une boucle ################################
# Boucle pour chaque année de 2020 à 2050
for year in range(2020, 2051):
    globals()[f'df_{year}'] = df[df['work_year'] == year]

globals() te permet d'accéder à l'espace de noms global (où les variables sont stockées).
f'df_{year}' crée une chaîne de caractères dynamique pour chaque année (par exemple, "df_2020", "df_2021", etc.).
En utilisant cette approche, tu crées directement des variables comme df_2020, df_2021, etc., et tu leur attribues les valeurs filtrées.





























