import pandas as pd
from google.colab import drive
import matplotlib.pyplot as plt
import textwrap

drive.mount('/content/drive', force_remount=True)
entree_xml = "/content/drive/MyDrive/"

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

sources = pd.read_csv (entree_xml+'/sources.csv', encoding='utf8',sep=';')
sources.rename(columns={'legibility': 'Lisibilité'}, inplace=True)
sources_legibility = sources.groupby('administrative_event')['Lisibilité'].value_counts().unstack()

sources_legibility.rename(columns={'Easily Legible Text': 'Lisible','Moderately Legible Text': 'Modérement lisible'}, inplace=True)
print(sources_legibility)

# set font size
plt.rcParams.update({'font.size': 16})

# create horizontal bars
sources_legibility.plot.barh(figsize=(12,6))

# moving legend
plt.legend(loc='center right')

# naming the x-axis
plt.xlabel('Nombre de documents')

# naming the y-axis
plt.ylabel('Type de sources')

# plot title
#plt.title('Lisibilté')

# wrapping labels
ax = plt.axes()
def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_yticklabels(labels, rotation=90, ma='center')
wrap_labels(ax, 10)

# save the plot as image
plt.savefig(entree_xml+'/sources_legibility.png')

# function to show the plot (need to be after saving)
plt.show()