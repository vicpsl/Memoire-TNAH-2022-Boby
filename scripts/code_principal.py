'''
Projet réalisé par JV BOBY dans le cadre du stage de fin d'étude
Mémoire pour le diplôme de master « Technologies numériques appliquées à l’histoire »
2022

La déportation des Roms en Transnistrie, 1942-1944.
Étude de l’appariement de listes de déportés.

Réalisé avec Python Record Linkage Toolkit:
A toolkit for record linkage and duplicate detection in Python
https://github.com/J535D165/recordlinkage

'''




# modules import  
import pandas as pd
from google.colab import drive
from random import randrange
#!pip install Levenshtein
#import Levenshtein as lev
#from Levenshtein import *
!pip install fuzzywuzzy
from fuzzywuzzy import fuzz, process
import itertools
from operator import itemgetter
import unicodedata
import numpy as np
import re
!pip install "textdistance[extras]"
import textdistance

!pip install py_stringmatching
from py_stringmatching import MongeElkan as ME
from py_stringmatching import JaroWinkler as JW

# prepare alias for the textdistance modules
txtlev = textdistance.levenshtein.normalized_similarity
txtdlev = textdistance.damerau_levenshtein.normalized_similarity
txtjw = textdistance.jaro_winkler.normalized_similarity
txtedtx = textdistance.editex.normalized_similarity
txtsw = textdistance.smith_waterman.normalized_similarity
txtjccd = textdistance.jaccard.normalized_similarity
pyMEJW = ME(sim_func=JW().get_sim_score).get_raw_score

!pip install recordlinkage
import recordlinkage
from recordlinkage.index import SortedNeighbourhood
from recordlinkage.index import Block

from sklearn.model_selection import train_test_split

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

# creating the "df_spelling" dictionary used below to expand name abbreviations (C-tin to Constantin)
df_spelling = pd.read_excel (entree_xml+'/spelling.xlsx', usecols=[0,1], dtype={'Entry':'object', 'Correct_Spelling': 'object'})
df_spelling = pd.Series(df_spelling.Correct_Spelling.values,index=df_spelling.Entry).to_dict()
df_spelling = {rf"\b{k}\b" : v for k, v in df_spelling.items()}

# transform newborns age to 0
def Convert_Age(x):
  if not x:
    return np.nan
  try:
    return '0' if '[' in str(x) else str(x)   
  except:        
    return np.nan

# load original data
ushmm = pd.read_csv (entree_xml+'/personUSHMM.csv', encoding='utf8',sep=';', index_col=False, converters={'peNumberAge-Age': Convert_Age, 'peNumberAge-Residence Age': Convert_Age}, dtype = str)

# load latest study files
individuals = pd.read_csv (entree_xml+'/individuals.csv', encoding='utf8',sep=';', usecols = ['id','gender','first_name','last_name','judet','household', 'relative', 'birthyear','marital_status'],  dtype = str)
census = pd.read_csv (entree_xml+'/census.csv', encoding='utf8',sep=';', usecols = ['individual','ushmm_id','line','digital_file','source','residence_territorial_entity'],  dtype = str)
deportation = pd.read_csv (entree_xml+'/deportation.csv', encoding='utf8',sep=';', usecols = ['individual','ushmm_id','line','digital_file','source','residence_territorial_entity'], dtype = str)
census = pd.merge(individuals, census, left_on=['id'], right_on=['individual'], how='left').drop('individual', axis=1)
deportation = pd.merge(individuals, deportation, left_on=['id'], right_on=['individual'], how='left').drop('individual', axis=1)

# filter records added manually
census = census[census['ushmm_id'].notna()]
census = census[~census['ushmm_id'].str.contains('n')]
deportation = deportation[deportation['ushmm_id'].notna()]
deportation = deportation[~deportation['ushmm_id'].str.contains('n')]

# sort dataframes in individual appearing order (by id)
census = census.sort_values(by=['ushmm_id'])
deportation = deportation.sort_values(by=['ushmm_id'])

# save latest study files
census.to_csv(entree_xml+'census_latest.csv', encoding="utf_8_sig", sep = ';', index=False)
deportation.to_csv(entree_xml+'deportation_latest.csv', encoding="utf_8_sig", sep = ';', index=False)

# get relevant original data and merge, suppress study equivalents
mask_ushmm = ushmm[['PersonId','Gender','FirstName','LastName','peNumberAge-Age','peDateBirth-Date of Birth','peFamilyRelationship-Family Relationship','peMaritalStatus-Marital Status','pePlaceResidence-Residence']]
ushmm_census = pd.merge(census, mask_ushmm, left_on=['ushmm_id'], right_on=['PersonId'], how='left').drop(['PersonId','gender','first_name','last_name','birthyear','relative','marital_status'], axis=1)
ushmm_deportation = pd.merge(deportation, mask_ushmm, left_on=['ushmm_id'], right_on=['PersonId'], how='left').drop(['PersonId','gender','first_name','last_name','birthyear','relative','marital_status'], axis=1)

# sort dataframes in individual appearing order (by id)
ushmm_census = ushmm_census.sort_values(by=['ushmm_id'])
ushmm_deportation = ushmm_deportation.sort_values(by=['ushmm_id'])

# save latest study records with original data
ushmm_census.to_csv(entree_xml+'original_census_latest.csv', encoding="utf_8_sig", sep = ';', index=False)
ushmm_deportation.to_csv(entree_xml+'original_deportation_latest.csv', encoding="utf_8_sig", sep = ';', index=False)

def getlatestindividuals():

  individuals = pd.read_csv (entree_xml+'/individuals.csv', encoding='utf8',sep=';', usecols = ['id','gender','first_name','last_name','judet','household', 'relative', 'birthyear'],  dtype = str)
  census = pd.read_csv (entree_xml+'/census.csv', encoding='utf8',sep=';', usecols = ['individual','ushmm_id','line','digital_file','source','residence_territorial_entity'],  dtype = str)
  deportation = pd.read_csv (entree_xml+'/deportation.csv', encoding='utf8',sep=';', usecols = ['individual','ushmm_id','line','digital_file','source','residence_territorial_entity'], dtype = str)
  individuals_latest = pd.merge(individuals, census, left_on=['id'], right_on=['individual'], how='left').drop('individual', axis=1)
  
  individuals_latest['source_type'] = np.where(individuals_latest['ushmm_id'].notnull(), 'census', np.nan)
  deportation.rename({'individual': 'id'}, axis=1, inplace=True)
  
  mapping = deportation.set_index('id')

  for c in ['ushmm_id','line','digital_file','source','residence_territorial_entity']:
    individuals_latest[c] = individuals_latest[c].fillna(individuals_latest['id'].map(mapping[c]))
  
  #individuals_latest.loc[individuals_latest['ushmm_id'].notna() & individuals_latest['source_type'].isnull(), 'source_type'] = 'deportation'
  individuals_latest['source_type'] = np.where((individuals_latest['ushmm_id'].notnull()) & (individuals_latest['source_type'] != 'census'), 'deportation', individuals_latest['source_type'])
  
  individuals_latest.to_csv(entree_xml+'individuals_latest.csv', encoding="utf_8_sig", sep = ';', index=False)

  return print('individuals_latest.csv ready')
  
 
def Change_Column_Names(df, dict):
  """Function changing dataframe column labels to the standard names set out in the file standard_columns.xlsx
  :param df (dataframe): dataframe built from the source
  :param dict (dict): the dic created from the file standard_columns.xlsx
  :returns: df with standard column names that will be used in the code and outputs
  """
  use_cols = list(dict.keys())
  #print(use_cols)
  #print(df.columns)
  for y in df.columns:
    if y not in use_cols:
      #print('not in use_cols '+ y)
      df = df.drop(y, axis=1)
  df = df.rename(columns = dict)
  #print(df.info())
  return df
    
def Compute_Birthyear(df, doc_year: int, file):
  """Function converting age, where available, to birth year
  :param df (dataframe): dataframe built from the source
  :param doc_year (YYYY-format integer): the date of the archive
  :param file (string): name of the file (without extension) for which we compute the birth year
  :returns: df with birthyear field computed from age if available
  """
  if 'birthyear' in df.columns and not 'age' in df.columns:
    print('The file '+ file + '.csv already has a birthyear field and no age field, formatting birthyear as str.')
    df['birthyear'] = df['birthyear'].apply(lambda x: str(int(x)) if not pd.isnull(x) else np.nan)
    return df
  elif (df.age.notnull() & df.birthyear.isnull()).sum() >=1:
    print('Calculating birthyear for :' + file)
    df['birthyear'] = df.apply(lambda row: int(doc_year) - int(row['age']) if (pd.isnull(row['birthyear']) and pd.notna(row['age'])) else row['birthyear'], axis=1)
    df['birthyear'] = df['birthyear'].apply(lambda x: str(int(x)) if not pd.isnull(x) else np.nan)
    df = df.astype(str)
  return df

def Normalise(df):
  """Function normalising input data (lowercase, removing diacritics), to enhance future matching and string comparisons
    :param df (dataframe): dataframe built from the source
    :returns: lowercase, diacritic-free standardised dataframe
  """
  cols = df.select_dtypes(include=[object]).columns
  df[cols] = df[cols].apply(lambda x: x.str.lower().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'), axis=0)
  return df

def Appellation(df, cols):
  """Function capturing and isolating nicknames intriduced by "zis"/"zisa" (Babu zis Arapu / Niţu zis Vasile), to enhance future matching and string comparisons
    :param df (dataframe): dataframe built from the source
    :param cols (columns, aka 'series' of dataframe to normalise): here, first & last names
    :returns: cleansed df for first & last names columns
  """
  for col in cols:
    appellation = df[col].str.extract(r'( zis.*$)')
    df.insert(loc = df.columns.get_loc(col) + 1 , column = col.replace('name','') + '_appelation', value = appellation[0])
    df[col] = df[col].str.replace(r'( zis.*$)', '')
  return df

def Remove_Irrelevant_Char_Name(df, cols):
  """Function normalising input data (removing characters excluded by the Regex rules below), to enhance future matching and string comparisons
    :param df (dataframe): dataframe built from the source
    :param cols (columns, aka 'series' of dataframe to normalise): here, first & last names
    :returns: cleansed df for first & last names columns
  """
  irrelevant_regex = re.compile(r'[^\-\?\.a-zA-Z0-9\s]') # remove any character that is not - or ? or a dot followed by text+space
  multispace_regex = re.compile(r'\s\s+') # remove multiple spaces
  
  for col in cols:
    df[col]=df[col].str.replace(irrelevant_regex, ' ').str.replace(multispace_regex, ' ')
    df[col]=df[col].str.replace(re.compile(r'\.(?=[a-z])'), '. ') # dots immediately followed by dot are replaced by a space  #modifier pour les .. ...
    
  return df

def Dot(df, var):
  """Function removing remaining dots, those following letters & followed by a space or end of string (dots used after abbreviations as in Nicolae N. Iancu)
    :param df (dataframe): dataframe built from the source
    :param var (columns, aka 'series' of dataframe to normalise): here, all segmented first & last names
    :returns: df with dot-cleansed first & last names columns
  """
  dot_regex = re.compile(r'(?<=[a-zA-Z])\.{1}?(?=\ |$)') # identifies a unique dot immediately following letters and followed by space or end of string.
  for col in var:
    df[col]=df[col].str.replace(dot_regex,'')
  return df


def List_Firstnames(df):
  """Function listing an individual firstnames
    :param df (dataframe): dataframe built from the source
    :returns: df with 2 new fields, one storing list of firstnames, another the number of these firstnames
  """
  df['list_firstnames'] = df['firstname'].apply(lambda x: x.split(' ') if np.all(pd.notnull(x)) else np.nan)
  df['firstnames_count'] = df['firstname'].apply(lambda x: len(str(x).split(' ')) if np.all(pd.notnull(x)) else 0)
  return df

def combinations(left, right):
  """Function looping throught available firstnames for an individual and returning the best match
    :param left (string): list of firstnames from the left file of the match
    :param right (string): list of firstnames from the right file of the match
    :returns: best firstname match as a string
  """
  matches = []
  best = []
  if left and right:
    leftlist = list(map(str, left.strip().split(" ")))
    rightlist = list(map(str, right.split(" ")))
    for seq_1 in leftlist:
      for seq_2 in rightlist:
        if len(seq_1)>1 and len(seq_2)>1:
          if seq_1 == seq_2:
            matches.append((seq_1, seq_2, 1))
            leftlist.remove(seq_1)
            rightlist.remove(seq_2)
          else: 
            matches.append((seq_1, seq_2, round(txtlev(seq_1, seq_2),2)))
        elif (len(seq_1) * len(seq_2))>1:
          if seq_2.startswith(seq_1) or seq_1.startswith(seq_2):
            matches.append((seq_1, seq_2, 0.5))
    if matches:
      best = [max(matches,key=itemgetter(2))[0], max(matches,key=itemgetter(2))[1],max(matches,key=itemgetter(2))[2]]
    else:
      best= ['None','None',0]
  else:
    best= ['None','None',0]
  return best


def Split_Names(df, cols):
  """Function splitting first & last names, to enhance future matching and string comparisons
    :param df (dataframe): dataframe built from the source
    :param cols (columns, aka 'series' of dataframe to normalise): here, first & last names
    :returns: df with segmented first & last names columns
  """
  # keep copy of original firstname (needed for other functions)
  df['original_first'] = df['firstname']
  # split names in incremental columns firstname_1, firstname_2, firstname_i (i last firstname index)
  for col in cols:
    df_extraction = df[col].str.split(' ', expand=True)
    df_extraction.columns = [col+'_'+str(i+1) for i in df_extraction.columns]
    df[col] = df_extraction.iloc[:,0]
    x = list(df_extraction.columns).index(max(list(df_extraction.columns)))
    df = pd.concat([df, df_extraction.iloc[:,1:x+1]], axis=1)
    df = df.reindex(sorted(df.columns), axis=1)
    return df

#ex dot place


def Fullname(df, var):
  """Function removing remaining dots, those following letters & followed by a space or end of string (dots used after abbreviations as in Nicolae N. Iancu)
    :param df (dataframe): dataframe built from the source
    :param var (columns, aka 'series' of dataframe to normalise): here, all segmented first & last names
    :returns: df with dot-cleansed first & last names columns
  """
  last_column = var[-1]
  last_index = df.columns.get_loc(last_column)
  fullname = df[var].apply(lambda x: ' '.join(sorted(x.dropna().astype(str))), axis=1)
  df.insert(loc = last_index + 1 , column = 'fullname', value = fullname )
  
  return df

def Derive_Gender(df):
  """Function deriving missing genders from the firstname-gender table gender.csv
  :param df (dataframe): dataframe built from the source
  :returns: df with derived genders from firstnames
  """
  # loading the gender mapping file
  firstnames_genders = pd.read_csv (entree_xml+'/gender.csv', encoding='utf8',sep=';', index_col=False, dtype = str)
  
  # filling blank values in original gender to 'undertermined'
  df['gender'] = df['gender'].fillna('undetermined')

  df['first_firstname'] = df['firstname'].apply(lambda x: x.split(' ')[0] if np.all(pd.notnull(x)) else np.nan)  
  # create a mapping dictionary
  mapping = dict(firstnames_genders[['firstname', 'derived_gender']].values)
  
  # assign the mapped genders based on firstname in the source file and filling missing values as empty
  df['derived_gender'] = df['first_firstname'].map(mapping)
  df['derived_gender'].fillna(np.nan, inplace=True)
  
  # suppress derived genders not in alignment with relative status
  conditions = [(df['relative'].isin(['husband', 'brother','son','father','grandfather','frate','fratii','fiu']) & df['derived_gender'].eq('female')),
                (df['relative'].isin(['wife', 'spouse','daughter','concubine','mother','sister','granddaughter','grandmother','aunt','fiica','sora']) & df['derived_gender'].eq('male'))]
  choices = ['undetermined','undetermined']
  
  # where derived genders contradict relative status, set to undetermined, otherwise leave derived value
  df['derived_gender'] = np.select(conditions, choices, df['derived_gender'])
  
  # where derived genders empty, fill with original genders (this wil lalso set 'undetermined' to NaN values where applicable)
  df['derived_gender'] = df['derived_gender'].fillna(df['gender'])
    
  # resetting type as str
  df[['gender','derived_gender']] = df[['gender','derived_gender']].astype(str)
  
  # removing first_firstname temporary column
  df.drop(columns=['first_firstname'], inplace=True)

  # congruency control between original genders and derived genders
  #print('\n "gender derived = gender" congruence : \n', df.groupby(['gender', 'derived_gender'])['derived_gender'].count())

  return df

def Family_Count(df):
  """Function counting the number of individuals within a household (family) id
  :param df (dataframe): dataframe built from the source
  :returns: df with family count column
  """ 
  df_members = df.groupby('household', dropna='True').size().reset_index(name='members_count')
  df = df.merge(df_members, on='household', how='left')
 
  return df
'''
def Family_Firstnames(df):
  """Function listing the first firstname of individual within a household (family) id
  :param df (dataframe): dataframe built from the source
  :returns: df with family firstnames column
  """ 
  df_fam_firstnames = df.groupby('household', dropna='True')['firstname'].apply(list).reset_index(name='fam_firstnames')
  df_fam_firstnames['fam_firstnames'] = df_fam_firstnames['fam_firstnames'].apply(lambda x: sorted(str(i) for i in x))
  df_fam_firstnames['fam_firstnames'] = df_fam_firstnames['fam_firstnames'].apply(lambda x: sorted(str(i) for i in x))
  df_fam_firstnames['fam_firstnames'] = df_fam_firstnames['fam_firstnames'].apply(lambda x: ' '.join(x))
  df = df.merge(df_fam_firstnames, on='household', how='left')
 
  return df

def Neighbours(df):
  """Function listing the first firstname of persons immediately before and after within the registry
  :param df (dataframe): dataframe built from the source
  :returns: df with immediate neighbour firstnames column
  """ 
  df['neighbours'] = df['firstname'].shift(1) + ' ' + df['firstname'].shift(-1)

  return df

def Family_Firstnames_Init(df):
  """Function listing the initials (main firstnames) of individuals within a household (family) id
  :param df (dataframe): dataframe built from the source
  :returns: df with family initials column
  """
  df_fam_firstnames_init = df.groupby('household', dropna='True')['firstname'].apply(list).reset_index(name='fam_firstnames_init')
  df_fam_firstnames_init['fam_firstnames_init'] = df_fam_firstnames_init['fam_firstnames_init'].apply(lambda x: sorted(str(i)[0] for i in x))
  df = df.merge(df_fam_firstnames_init, on='household', how='left')
  
  return df
'''


# entirely prepare the 2 input files for the study, thanks to the above functions
def Prepare(file_a,year_a: int,file_b, year_b: int):
  """Function preparing 2 files the user would like to standardise and compare, saves them as (provided file_a).csv & (provided file_b).csv, returns them as "df_a" & "df_b" for further processes
    :param file_a (string): name without extension of the first file to be used in the study
    :param year_a (int): production year of the first file (to compute birthyear from the document's ages if any)
    :param file_b (string): name without extension of the second file to be used in the study
    :param year_a (int): production year of the second file (to compute birthyear from the document's ages if any)
    :returns: file_a (as df_a) and file_b (as df_b) cleansed and prepared for the next steps
  """
  # load the files
  df_a = pd.read_csv (entree_xml+'/'+ file_a + '.csv', encoding='utf8',sep=';', index_col=False, converters={'peNumberAge-Age': Convert_Age, 'peNumberAge-Residence Age': Convert_Age}, dtype = str)
  df_b = pd.read_csv (entree_xml+'/'+ file_b + '.csv', encoding='utf8', sep=';', index_col=False, converters={'peNumberAge-Age': Convert_Age, 'peNumberAge-Residence Age': Convert_Age}, dtype = str)
  
  # creating the "standard_cols_dict" dictionary used by Change_Column_Names(df_a,standard_cols_dict) to standardise column labels
  df_standard_cols = pd.read_excel (entree_xml+'/standard_columns2.xlsx', usecols=[0,1], dtype={'old_column_name':'object', 'standard_column_name': 'object'})
  standard_cols_dict = pd.Series(df_standard_cols.standard_column_name.values,index=df_standard_cols.old_column_name).to_dict()

  #print('\df_a head\n',df_a.head(10))
  #print('\df_b head\n',df_b.head(10))

  # renaming column labels by function Change_Column_Names(df_a,standard_cols_dict) with the standard column names dictionary
  df_a = Change_Column_Names(df_a,standard_cols_dict)
  df_b = Change_Column_Names(df_b,standard_cols_dict)

  # sorting by individuals appearing order (by id)
  df_a = df_a.sort_values(by=['ushmm_id'])
  df_b = df_b.sort_values(by=['ushmm_id'])

  #print('\df_a head\n',df_a.head(10))
  #print('\df_b head\n',df_b.head(10))
  
  Compute_Birthyear(df_a,year_a, file_a)
  Compute_Birthyear(df_b,year_b, file_b)

  df_a = Normalise(df_a)
  df_b = Normalise(df_b)

  # expanding name abbreviations (C-tin to Constantin) from the df_spelling dictionary, built in preambule above
  firstlast_cols=['firstname','lastname']
  df_a.loc[:, firstlast_cols] = df_a[firstlast_cols].replace(df_spelling, regex=True)
  df_b.loc[:, firstlast_cols] = df_b[firstlast_cols].replace(df_spelling, regex=True)

  df_a = Appellation(df_a, firstlast_cols)
  df_b = Appellation(df_b, firstlast_cols)

  df_a = df_a.loc[:,~df_a.columns.str.contains('appelation', case=False)]
  df_b = df_b.loc[:,~df_b.columns.str.contains('appelation', case=False)] 

  df_a = Remove_Irrelevant_Char_Name(df_a,firstlast_cols)
  df_b = Remove_Irrelevant_Char_Name(df_b,firstlast_cols)

  df_a = Dot(df_a, var = [col for col in df_a.columns if 'name' in col] )
  df_b = Dot(df_b, var = [col for col in df_b.columns if 'name' in col] )


  #df_a = List_Firstnames(df_a)
  #df_b = List_Firstnames(df_b)

  df_a = Derive_Gender(df_a)
  df_b = Derive_Gender(df_b)

  df_a = Split_Names(df_a,firstlast_cols)
  df_b = Split_Names(df_b,firstlast_cols)
  #ex dot place
  
  df_a_cols_full = [col for col in df_a.columns if ('name' in col and not 'names' in col)]
  #df_a_cols_full = ['firstname','lastname']
  df_a = Fullname(df_a, var = df_a_cols_full)
  df_b_cols_full = [col for col in df_b.columns if ('name' in col and not 'names' in col)]
  #df_b_cols_full = ['firstname','lastname']
  df_b = Fullname(df_b, var = df_b_cols_full)

  df_a = Family_Count(df_a)
  df_b = Family_Count(df_b)

  #df_a = Family_Firstnames(df_a)
  #df_b = Family_Firstnames(df_b)

  #df_a = Neighbours(df_a)
  #df_b = Neighbours(df_b)

  #df_a = Family_Firstnames_Init(df_a)
  #df_b = Family_Firstnames_Init(df_b)

  


  # save latest study records (original data) prepared
  df_a.to_csv(entree_xml + file_a + '_prepared.csv', encoding="utf_8_sig", sep = ';', index=False)
  df_b.to_csv(entree_xml + file_b + '_prepared.csv', encoding="utf_8_sig", sep = ';', index=False)
  
  
  return df_a, df_b

def Test_Scores(df):
  """Function calculating a set of string similarities between 2 files already matched
    :param df (string): name without extension of the ids mapping (ushmm_id) file between the 2 matched files
    :returns: df with a set of string similarities between 2 files already matched
  """
  # load the persons ids mapping file
  manual_matches = pd.read_csv (entree_xml + '/' + df + '.csv', encoding='utf8',sep=';', index_col=False, dtype = str)
  
  # load the files
  original_census_latest_prepared = pd.read_csv (entree_xml + '/original_census_latest_prepared.csv', encoding='utf8',sep=';', index_col=False, dtype = str)
  original_deportation_latest_prepared = pd.read_csv (entree_xml + '/original_deportation_latest_prepared.csv', encoding='utf8',sep=';', index_col=False, dtype = str)

  # merge prepared original data based on the mapping ids file
  scores = pd.merge(manual_matches, original_census_latest_prepared, left_on=['census'], right_on=['ushmm_id'], how='left').drop('census', axis=1)
  scores = pd.merge(scores, original_deportation_latest_prepared, left_on=['deportation'], right_on=['ushmm_id'], how='left').drop('deportation', axis=1)

  # select rows where data is available and persons not having firstname = lastname (to evaluate matches where first and last names are reversed)
  names_notna = scores[scores['firstname_x'].notnull() & scores['lastname_y'].notnull() & scores['firstname_y'].notnull() & scores['lastname_x'].notnull() & (scores['firstname_x'] not in scores['lastname_x'])][['ushmm_id_x','firstname_x','lastname_x','firstname_y','lastname_y']]
  
  # create reversed first/last names indicator based on 80% levenshstein match
  names_notna['reversed'] =  names_notna[['firstname_x','lastname_x','firstname_y','lastname_y']].apply(lambda x: 'Y' if (txtlev(x[0],x[3]) >= 0.8 and txtlev(x[2],x[1]) >= 0.8) else 'N', axis=1)

  # only keep ids for the selection and change mergekey name (to remove it in next step)
  names_notna.drop(['firstname_x','lastname_x','firstname_y','lastname_y'], axis=1, inplace=True)
  names_notna.rename({'ushmm_id_x': 'temp_id'}, axis=1, inplace=True)
  
  # append reversed indicator to the main score file, fillna with indicator N (names not reversed)
  scores = pd.merge(scores, names_notna, left_on=['ushmm_id_x'], right_on=['temp_id'], how='left').drop('temp_id', axis=1)
  scores['reversed'] = scores['reversed'].fillna('N')
   
  # get best match between individuals firstnames

  scores['firstnames_fuzzy_tok_set'] = scores[['original_first_x','original_first_y']].apply(lambda x: combinations(x[0],x[1])[0:2] if(np.all(pd.notnull(x[0])) and np.all(pd.notnull(x[1]))) else None, axis=1) # if (' ' in x[0] or ' ' in x[1]) else None
  scores['best_firstname_x'] = scores['firstnames_fuzzy_tok_set'].apply(lambda x: x[0] if x else None)
  scores['best_firstname_y'] = scores['firstnames_fuzzy_tok_set'].apply(lambda x: x[-1] if x else None) #isinstance(x, list)
  
  # calculate set of scores on available data (field by field : atomic comparison)

  scores['Additional_FirstNames_x'] = scores[['best_firstname_x','original_first_x']].apply(lambda x: ' '.join([str(i) for i in x[1].split() if str(i) != x[0]]) if(np.all(pd.notnull(x[0])) and np.all(pd.notnull(x[1]))) else None, axis=1)
  scores['Additional_FirstNames_y'] = scores[['best_firstname_y','original_first_y']].apply(lambda x: ' '.join([str(i) for i in x[1].split() if str(i) != x[0]]) if(np.all(pd.notnull(x[0])) and np.all(pd.notnull(x[1]))) else None, axis=1)

  # scores['sc_Family_Firstnames'] = scores[['fam_firstnames_x','fam_firstnames_y']].apply(lambda x: round(pyMEJW((' '.join(map(str, x[0]))).split(),(' '.join(map(str, x[1]))).split()),2) if(np.all(pd.notnull(x[0])) and np.all(pd.notnull(x[1]))) else None, axis=1)

  scores['sc_best_firstname_lev'] = scores[['best_firstname_x','best_firstname_y']].apply(lambda x: round(txtlev(x[0],x[1]),2) if(np.all(pd.notnull(x[0])) and np.all(pd.notnull(x[1]))) else None, axis=1)
  scores['sc_firstname_lev'] = scores[['firstname_x','firstname_y']].apply(lambda x: round(txtlev(x[0],x[1]),2) if(np.all(pd.notnull(x[0])) and np.all(pd.notnull(x[1]))) else None, axis=1)
  scores['sc_lastname_lev'] = scores[['lastname_x','lastname_y']].apply(lambda x: round(txtlev(x[0],x[1]),2) if(np.all(pd.notnull(x[0])) and np.all(pd.notnull(x[1]))) else None, axis=1)

  scores['sc_firstname_dlev'] = scores[['firstname_x','firstname_y']].apply(lambda x: round(txtdlev(x[0],x[1]),2) if(np.all(pd.notnull(x[0])) and np.all(pd.notnull(x[1]))) else None, axis=1)
  scores['sc_lastname_dlev'] = scores[['lastname_x','lastname_y']].apply(lambda x: round(txtdlev(x[0],x[1]),2) if(np.all(pd.notnull(x[0])) and np.all(pd.notnull(x[1]))) else None, axis=1)

  scores['sc_firstname_jw'] = scores[['firstname_x','firstname_y']].apply(lambda x: round(txtjw(x[0],x[1]),2) if(np.all(pd.notnull(x[0])) and np.all(pd.notnull(x[1]))) else None, axis=1)
  scores['sc_lastname_jw'] = scores[['lastname_x','lastname_y']].apply(lambda x: round(txtjw(x[0],x[1]),2) if(np.all(pd.notnull(x[0])) and np.all(pd.notnull(x[1]))) else None, axis=1)

  scores['sc_firstname_sw'] = scores[['firstname_x','firstname_y']].apply(lambda x: round(txtsw(x[0],x[1]),2) if(np.all(pd.notnull(x[0])) and np.all(pd.notnull(x[1]))) else None, axis=1)
  scores['sc_lastname_sw'] = scores[['lastname_x','lastname_y']].apply(lambda x: round(txtsw(x[0],x[1]),2) if(np.all(pd.notnull(x[0])) and np.all(pd.notnull(x[1]))) else None, axis=1)

  scores['sc_fullname_lev'] = scores[['fullname_x','fullname_y']].apply(lambda x: round(txtlev(x[0],x[1]),2) if(np.all(pd.notnull(x[0])) and np.all(pd.notnull(x[1]))) else None, axis=1)
  scores['sc_fullname_dlev'] = scores[['fullname_x','fullname_y']].apply(lambda x: round(txtdlev(x[0],x[1]),2) if(np.all(pd.notnull(x[0])) and np.all(pd.notnull(x[1]))) else None, axis=1)
  scores['sc_fullname_jw'] = scores[['fullname_x','fullname_y']].apply(lambda x: round(txtjw(x[0],x[1]),2) if(np.all(pd.notnull(x[0])) and np.all(pd.notnull(x[1]))) else None, axis=1)
  scores['sc_fullname_sw'] = scores[['fullname_x','fullname_y']].apply(lambda x: round(txtsw(x[0],x[1]),2) if(np.all(pd.notnull(x[0])) and np.all(pd.notnull(x[1]))) else None, axis=1)
  # calculating Monge-Elkan only for fullname as it has the property of processing multiword comparisons
  scores['sc_fullname_ME'] = scores[['fullname_x','fullname_y']].apply(lambda x: round(pyMEJW(x[0].split(),x[1].split()),2) if(np.all(pd.notnull(x[0])) and np.all(pd.notnull(x[1]))) else None, axis=1)

  # calculating mean of atomic scores between first and lastnames results (except Monge-Elkan as indicated)
  scores['sc_lev_avg'] = round(scores[['sc_firstname_lev', 'sc_lastname_lev']].mean(axis=1),2)
  scores['sc_dlev_avg'] = round(scores[['sc_firstname_dlev', 'sc_lastname_dlev']].mean(axis=1),2)
  scores['sc_jw_avg'] = round(scores[['sc_firstname_jw', 'sc_lastname_jw']].mean(axis=1),2)
  scores['sc_sw_avg'] = round(scores[['sc_firstname_sw', 'sc_lastname_sw']].mean(axis=1),2)

  
  # converting to float
  float_scores_cols = [col for col in scores if col.startswith('sc_')]
  scores[float_scores_cols] = scores[float_scores_cols].astype(float)
  
  # save latest study records (original data) prepared & scored
  scores.to_csv(entree_xml + '/scores_test.csv', encoding="utf_8_sig", sep = ';', decimal=",", index=False)

  return scores

def Record_Linkage(file_a, file_b):
  """Function performing Record Linkage Matching between (provided file_a).csv & (provided file_b).csv, returning the set of results for analysis
    :param file_a (string): name without extension of the first file to be used in Record Linkage
    :param file_b (string): name without extension of the second file to be used in Record Linkage
    :returns: 
  """
  # test whether the prepared files exist
  try:
    df_a = pd.read_csv (entree_xml+'/'+ file_a + '.csv', encoding='utf8',sep=';', index_col=False, dtype = str)
  except:
    return print('The file '+ file_a + 'does not exist, please prepare the file first and retry')
  try:
    df_b = pd.read_csv (entree_xml+'/'+ file_b + '.csv', encoding='utf8',sep=';', index_col=False, dtype = str)
  except:
    return print('The file '+ file_b + 'does not exist, please prepare the file first and retry')
  

  # defining blockkeys
  df_a['blockkey1'] = df_a['residence_territorial_entity']
  df_b['blockkey1'] = df_b['residence_territorial_entity']

  # other examples and ways of creating blocking key for the user information
  # defining combinations of fields
  # blockkey2_cols = ['residence_territorial_entity','birthyear','firstname','lastname']
  # creating the above fields combination
  # df_a['blockkey2'] = df_a[blockkey2_cols].fillna('').sum(axis=1)

  # definign a blockkey with the first 2 caracters of the territorial entity and the initial of the firstname and last caracter of firstname
  # df_a['blockkey3'] = df_a['residence_territorial_entity'].str[:2] + df_a['firstname'].str[:1] + df_a['firstname'].str[-1]
  
  # set up indexer for Recordlinkage
  indexer = recordlinkage.Index()
  #indexer.full() # full index, this exceeds Colab's memory capacity
  indexer.block(['blockkey1','derived_gender'],['blockkey1','derived_gender'])
  #indexer.block('digital_file','digital_file')
  #indexer.block('lastname','lastname')
  #indexer.block('derived_gender','derived_gender') # this alone exceeds Colab's memory capacity
  #indexer.block('relative','relative')
  #indexer.block('birthyear','birthyear')
  #indexer.add(SortedNeighbourhood(left_on='lastname', right_on='lastname', block_on=['blockkey1','derived_gender'], window=9))

  compare_cols = ['ushmm_id', 'birthyear', 'firstname', 'firstname_2', 'firstname_3', 'derived_gender', 'lastname', 'fullname', 'relative', 'household','residence_territorial_entity', 'source_id', 'blockkey1']
  df_a = df_a.filter(compare_cols)
  df_b = df_b.filter(compare_cols)

  #df_a.index = np.arange(len(df_a))
  #df_b.index = np.arange(len(df_b))

  df_a.set_index('ushmm_id', inplace=True, drop=True)
  df_b = df_b.set_index('ushmm_id')

  print('\n dfa_tomatch head\n', df_a.head(2))
  print('\n dfb_tomatch head\n', df_b.head(2))

  # index pairs
  print('\n dfa cols :\n', df_a.columns)
  print('\n dfb cols :\n', df_b.columns)

  candidate_pairs = indexer.index(df_a, df_b)
  print("candidate_pairs:", len(candidate_pairs))
  print('\ncandidate_pairs index : \n',candidate_pairs)

  
  # initialise class
  comp = recordlinkage.Compare()

  # initialise similarity measurement algorithms
  comp.string('firstname', 'firstname', method='levenshtein', threshold=0.70, missing_value=0, label='firstname')
  comp.string('firstname_2', 'firstname_2', method='levenshtein', threshold=0.70, missing_value=0, label='firstname2')
  comp.string('lastname', 'lastname', method='levenshtein', threshold=0.80, missing_value=0, label='lastname')
  comp.string('fullname', 'fullname', method='levenshtein', threshold=0.80, missing_value=0, label='fullname')
  #comp.string('firstname', 'lastname', method='levenshtein', threshold=0.85, label='firstname in last')
  #comp.string('lastname', 'firstname', method='levenshtein', threshold=1, label='lastname in first')
  comp.exact('birthyear', 'birthyear',  missing_value=0, label='birthyear')
  #comp.numeric('birthyear', 'birthyear', method='linear', offset=3, scale=3, missing_value=0.5, label='birthyear') #too much ressources needed, session crash
  comp.exact('household', 'household',  missing_value=0, label='household')
  comp.exact('derived_gender', 'derived_gender',  missing_value=0, label='gender')
  comp.exact('relative', 'relative', missing_value=0, label='relative')
  comp.exact('residence_territorial_entity', 'residence_territorial_entity', label='residence_territorial_entity')

  # the method .compute() returns the DataFrame with the feature vectors (1 for fields where the rule is fullfiled, 0 otherwise).
  features = comp.compute(candidate_pairs, df_a, df_b)
  
  # displaying results statistics
  # mean, standard, quantile, etc of results
  print(features.describe())
  # sum of pairs by number of attributes matching the rules
  print(features.sum(axis=1).value_counts().sort_index(ascending=False))
  # control view on screen
  print(features.head(10))
  
  # filtering potential matches where rules fullfilled total >=3 (can be adjusted) and both first & lastnames are matching the above criteria
  potential_matches = features[(features.sum(axis=1) >=3) & (features[['firstname','lastname']].sum(axis=1) >1)].reset_index()
  # add a column summing all results of the comparisons
  potential_matches['Score'] = potential_matches.loc[:, 'firstname':'residence_territorial_entity'].sum(axis=1)
  # control view on screen
  print(potential_matches.head(10))

  # RL toolkit only returns index and results, original data needs to be mapped back based on id
  # select columns to display with the results (usually the ones used in comparison rules)
  df_a_lookup = df_a[['fullname','firstname_2','birthyear','derived_gender','relative','household','residence_territorial_entity','source_id']].reset_index()
  df_b_lookup = df_b[['fullname','firstname_2','birthyear','derived_gender','relative','household','residence_territorial_entity','source_id']].reset_index()
  
  # adding the above columns to the results
  df_a_merge = pd.merge(potential_matches, df_a_lookup, left_on='ushmm_id_1',right_on='ushmm_id').drop('ushmm_id', axis=1)
  final_merge = pd.merge(df_a_merge, df_b_lookup, left_on='ushmm_id_2',right_on='ushmm_id').drop('ushmm_id', axis=1)
  # control view on screen
  print(final_merge.head(10))

  # exporting results to csv
  final_merge.to_csv(entree_xml+'/Record_Linkage.csv', encoding="utf_8_sig", sep = ';', decimal=",", index=False)
  
  #########
  # Unsupervised maching and evaluation of performance
  #########

  feature_vectors = comp.compute(candidate_pairs, df_a, df_b)
  
  # create training and test sets (a model should not be trained and tested on the same subset of the data)
  train, test = train_test_split(feature_vectors, test_size=0.25)

  # load the reference matching pairs
  manual_matches = pd.read_csv (entree_xml+'/census_deportation_ground_truth.csv', encoding='utf8',sep=';', index_col=False, dtype = str)
  manual_matches = manual_matches.set_index(['ushmm_id_x', 'ushmm_id_y']).index

  # load true pairs for the evalation of the test
  test_matches_index = test.index & manual_matches

  # built the K-mean classifier
  kmeans = recordlinkage.KMeansClassifier()

  # command for the model training
  result_kmeans = kmeans.learn(train)

  # build the predictions on the test set
  predictions = kmeans.predict(test)

  # prepare the confusion matrix (True/False positives and True/False negatives)
  confusion_matrix = recordlinkage.confusion_matrix(test_matches_index, predictions, len(test))

  # display the precision, recall and F-measure scores
  print('Précision (Precision) : ', recordlinkage.precision(confusion_matrix))
  print('Rappel (Recall) : ', recordlinkage.recall(confusion_matrix))
  print('F1-Score (F-Measure) : ', recordlinkage.fscore(confusion_matrix))

  return final_merge


#########
# Below the commands to run 
# 1) the preparation and normalisation of the USHMM files
# Columns naming conventions in the source files need to follow the standard_columns.xlsx values (either old or standard)
# 2) apply if needed a set of similarity scores on the prepared files
# 3) run record linkage matching on the "latest_prepared" set of files
#########

# command 1)
Prepare('original_census_latest',1942,'original_deportation_latest',1942)

# command 2)
scores_test = Test_Scores('manual_matches')

# command 3)
Record_Linkage('original_census_latest_prepared', 'original_deportation_latest_prepared')




