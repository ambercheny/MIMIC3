# All lab events in the database
LABEVENTS=pd.read_csv('./mimic_data/LABEVENTS.csv', parse_dates=True)
LABEVENTS1 = LABEVENTS.drop(['ROW_ID', 'SUBJECT_ID'], axis=1)
# Lab event ID
D_LABITEMS=pd.read_csv('./mimic_data/D_LABITEMS.csv', parse_dates=True)

""" filter the data """
# labevent includes the lab items we want to include for analysis
labevent=pd.read_csv('./mimic_data/labevent.txt', header=None, delimiter = "\t")
item_list = list(labevent[0])
# create a dataframe (df_lab) containing the IDs of warrante lab events
b = {}
for i in item_list:
    print(i)
    b[i] = pd.concat([D_LABITEMS.loc[D_LABITEMS['ITEMID']==i]],axis=0, ignore_index=True)

df_lab=pd.concat(b.values(), ignore_index=True)
len(df_lab)

# merge df_lab with LABEVENTS1 to filter the warranted lab events  
df_lab_merge= LABEVENTS1.merge(df_lab, on = "ITEMID", how='left')
# drop unnecessary columns
col = ["HADM_ID", "ITEMID", "VALUENUM"]
df_lab_merge =df_lab_merge[col]

""" now we use the dataframe to merge with training set IDs and testing set IDs """

#1. training set
training_dataset_label_task1=pd.read_csv('C:/Users/amber/Desktop/mimic/training_dataset_label_task1.csv')
HADM_list = list(training_dataset_label_task1["HADM_ID"])t
a = {}
for i in HADM_list:
#     print(i)
    a[i] = pd.concat([df_lab_merge.loc[df_lab_merge['HADM_ID']==i]],axis=0, ignore_index=True)
lab_processed = pd.concat(a.values(), ignore_index=True)

# merge with labels of training set
cols = training_dataset_label_task1.columns.tolist()
cols1 = ['LABEL', 'HADM_ID']
df_label = training_dataset_label_task1[cols1]
train_lab = df_label.merge(lab_processed, on = "HADM_ID", how = "left")
#train_lab.to_csv (r'C:\Users\amber\Desktop\mimic\processed\lab.csv', header=True)

#2. testing set
submission = pd.read_csv("C:/Users/amber/Desktop/mimic/submission_task1.csv")
HADM_list = list(submission["HADM_ID"])
a = {}
for i in HADM_list:
#     print(i)
    a[i] = pd.concat([df_lab_merge.loc[df_lab_merge['HADM_ID']==i]],axis=0, ignore_index=True)
submission_lab = pd.concat(a.values(), ignore_index=True)
#submission_lab.to_csv(r'C:\Users\amber\Desktop\mimic\processed\submission_lab.csv', header=True)

""" transform data"""
# make columns of the two tables are identical
cols = ['HADM_ID', 'ITEMID', 'VALUENUM', 'LABEL']
train_lab = train_lab[cols]
train_lab = train_lab.dropna()
train_lab['ITEMID'] = train_lab['ITEMID'].astype(int)

cols = ['HADM_ID', 'ITEMID', 'VALUENUM']
submission_lab = submission_lab[cols]
submission_lab['LABEL'] = ''
submission['HADM_ID'] = submission['HADM_ID'].astype(int)

# merge two tables and do pivot - making the column number of two tables identicle
merge_list = [task, submission]
df = pd.concat(merge_list, sort=False)
print(len(task))
print(len(submission))
print(len(df))

df=df.drop_duplicates()
df_pivot = pd.pivot_table(df, index='HADM_ID', columns='ITEMID', values='VALUENUM')
df_pivot = df_pivot.fillna(0)
df_pivot.head()

# split df_pivot back to two tables (training and testing) by IDs
# 1. training set
train_id = pd.read_csv('C:/Users/amber/Desktop/mimic/training_dataset_label_task1.csv')
train_df = train_id.merge(df_pivot, on = 'HADM_ID', how = 'left')
train_df.to_csv(r'C:/Users/amber/Desktop/mimic/processed/lab_task.csv', header=True)

# 2. testing set
submission_id = pd.read_csv('C:/Users/amber/Desktop/mimic/submission_task1.csv')
submission_id = submission_id.drop(columns = 'LABEL')
submission_id = submission_id.sort_values (by = 'HADM_ID')

sub_df = submission_id.merge(df_pivot, on = 'HADM_ID', how = 'left')
sub_df.to_csv(r'C:/Users/amber/Desktop/mimic/processed/lab_submission.csv', header=True)