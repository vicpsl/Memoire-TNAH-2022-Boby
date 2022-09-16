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

# source csv import (files previously prepared with the Prepare() function)
original_census_latest_prepared = pd.read_csv (entree_xml+'/original_census_latest_prepared.csv', encoding='utf8',sep=';', index_col=False, dtype = str)
original_deportation_latest_prepared = pd.read_csv (entree_xml+'/original_deportation_latest_prepared.csv', encoding='utf8',sep=';', index_col=False, dtype = str)

# remove irrelevant columns
original_census_latest_prepared = original_census_latest_prepared[['gender','derived_gender']]
original_deportation_latest_prepared = original_deportation_latest_prepared[['gender','derived_gender']]

# renaming to prepare graph labels
original_census_latest_prepared.rename(columns = {'gender': 'Genre', 'derived_gender': 'Genre inféré'}, inplace = True)
original_deportation_latest_prepared.rename(columns = {'gender': 'Genre', 'derived_gender': 'Genre inféré'}, inplace = True)

# get columns coverage in %
cs_gender_percent_coverage = original_census_latest_prepared.notnull().sum() * 100 / len(original_census_latest_prepared)
dpt_gender_percent_coverage = original_deportation_latest_prepared.notnull().sum() * 100 / len(original_deportation_latest_prepared)

# set % series as dataframes
census_gender_percent_coverage = pd.DataFrame(cs_gender_percent_coverage, columns=['Pourcentage']).reset_index()
deportation_gender_percent_coverage = pd.DataFrame(dpt_gender_percent_coverage, columns=['Pourcentage']).reset_index()

# insert source type to prepare pivot table
census_gender_percent_coverage.insert(1, 'Fichier', 'Census')
deportation_gender_percent_coverage.insert(1, 'Fichier', 'Déportation')

# rename columns to prepare pivot table
census_gender_percent_coverage.rename(columns = {'index': 'Attributs'}, inplace = True)
deportation_gender_percent_coverage.rename( columns = {'index': 'Attributs'}, inplace = True)

# merge the 2 dataframes
scope_coverage = pd.concat([ census_gender_percent_coverage , deportation_gender_percent_coverage])

# pivot table to create the graph
scope_content = scope_coverage.pivot_table( index = "Attributs", columns = "Fichier", values = "Pourcentage", aggfunc=np.sum , sort=False )

# set font size
plt.rcParams.update({'font.size': 16})

# plot % coverage by column in horizontal bars
scope_content.plot.barh(figsize=(12,6))

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
plt.savefig(entree_xml+'/derived_gender.png', bbox_inches='tight')

# function to show the plot (needs to be after saving)
plt.show()