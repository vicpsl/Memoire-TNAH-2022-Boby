# modules import
import pandas as pd
from google.colab import drive
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

# control imported data
print(ushmm.info())

# get columns coverage in %
percent_coverage = ushmm.notnull().sum() * 100 / len(ushmm)

# set font size
plt.rcParams.update({'font.size': 16})

# moving legend
#plt.legend(loc='center right')

# naming the x-axis
plt.xlabel('Taux de couverture')

# naming the y-axis
plt.ylabel('Attributs du fichier')

# plot % coverage by column in horizontal bars
percent_coverage.plot.barh(figsize=(6,15))

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
plt.savefig(entree_xml+'/ushmm_content.png', bbox_inches='tight')

plt.show()