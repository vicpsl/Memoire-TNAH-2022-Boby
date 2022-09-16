# modules import
import pandas as pd
from google.colab import drive
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# files path definition
drive.mount('/content/drive', force_remount=True)
entree_xml = "/content/drive/MyDrive/"

# set dataframe display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

# source csv import
ushmm = pd.read_csv (entree_xml+'/personUSHMM.csv', encoding='utf8',sep=';', index_col=False, dtype = str)
census = pd.read_csv (entree_xml+'/census.csv', encoding='utf8',sep=';', usecols = ['ushmm_id'],  dtype = str)
deportation = pd.read_csv (entree_xml+'/deportation.csv', encoding='utf8',sep=';', usecols = ['ushmm_id'], dtype = str)

# define irrelevant columns
drop_columns = ['Restricted','PersonType','Honorific','MiddleName','Suffix', 'Description','pePlaceBirth-Place of Birth','peNationality-Nationality','peMediaPrimary-Primary Media']

# filter individuals concerned by respectively the census and the deportation files
ushmm_census = ushmm[ushmm['PersonId'].isin(census['ushmm_id'])]
ushmm_deportation = ushmm[ushmm['PersonId'].isin(deportation['ushmm_id'])]

# control imported data
print(ushmm_census.info())
print(ushmm_deportation.info())

# group by, compute and label statictics
table = pd.DataFrame(ushmm_deportation.groupby(['peDateResidence-Date of Residence'], dropna=False).size(),columns=['Nombre']).reset_index()
table.rename(columns={'peDateResidence-Date of Residence':'Date Résidence'}, inplace = True)
table = table.sort_values(['Nombre'], ascending=[False])
print(table)


# remove irrelevant columns
ushmm_census.drop(drop_columns, axis=1, inplace=True)
ushmm_deportation.drop(drop_columns, axis=1, inplace=True)

# get columns coverage in %
cs_percent_coverage = ushmm_census.notnull().sum() * 100 / len(ushmm_census)
dpt_percent_coverage = ushmm_deportation.notnull().sum() * 100 / len(ushmm_deportation)
# set % series as dataframes
census_percent_coverage = pd.DataFrame(cs_percent_coverage, columns=['Pourcentage']).reset_index()
deportation_percent_coverage = pd.DataFrame(dpt_percent_coverage, columns=['Pourcentage']).reset_index()
# insert source type to prepare pivot table
census_percent_coverage.insert(1, 'Fichier', 'Census')
deportation_percent_coverage.insert(1, 'Fichier', 'Déportation')
# rename columns to prepare pivot table
census_percent_coverage.rename(columns = {'index': 'Attributs'}, inplace = True)
deportation_percent_coverage.rename(columns = {'index': 'Attributs'}, inplace = True)

# merge the 2 dataframes
scope_coverage = pd.concat([census_percent_coverage,  deportation_percent_coverage])
# pivot table to create the graph
scope_content = scope_coverage.pivot_table(index="Attributs", columns="Fichier", values="Pourcentage", aggfunc=np.sum, sort=False)

# set font size
plt.rcParams.update({'font.size': 16})

# plot % coverage by column in horizontal bars
scope_content.plot.barh(figsize=(6,15))

# moving legend
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

# naming the x-axis
plt.xlabel('Taux de couverture')

# naming the y-axis
plt.ylabel('Attributs du fichier')

# shortening labels (get text after '-' if any)
ax = plt.axes()
def wrap_labels(ax):
    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        labels.append(text.split('-')[1] if '-' in text else text)
    ax.set_yticklabels(labels, ma ='center')
wrap_labels(ax)

# desired format of the ticks, here '40%'
fmt = '%.0f%%'
# format the x axis
xticks = mtick.FormatStrFormatter(fmt)
ax.xaxis.set_major_formatter(xticks)

# reverse order y axis
ax.invert_yaxis()

# save the plot as image
plt.savefig(entree_xml+'/scope_content.png', bbox_inches='tight')

plt.show()