import os
import pandas as pd
from tdc.multi_pred import DTI
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

# 获取从Uniprot网站获得关于精神类疾病的Entry
md_datapath = "/home/cpua212/code/ylyao/MDDTI_YYL/Data/Mental_disorder/"
md_records = pd.read_csv(os.path.join(md_datapath, "source.csv"))
md_seq = list(md_records['Sequence'])
md_id = list(md_records['Entry'])
# Lenselink Dataset
lenselink = pd.read_pickle(os.path.join(md_datapath, 'data.pickle'))
print(lenselink.head())
print(len(list(lenselink['Molecule'].unique())))
print(len(list(lenselink['Protein'].unique())))
lenselink_md = lenselink[lenselink['Protein'].isin(md_seq)]
print(len(list(lenselink_md['Molecule'].unique())))
print(len(list(lenselink_md['Protein'].unique())))
lenselink_md = lenselink_md.rename(columns={'Molecule': 'Drug', 'Protein': 'Target'})
lenselink_md.to_csv(os.path.join(md_datapath, 'processed/Lenselink_md.csv'))
# train_set = lenslink_md[lenslink_md["temporal"]== "train"]
# test_set = lenslink_md[lenslink_md["temporal"]== "test"]
# valid_set = lenslink_md[lenslink_md["temporal"]=="valid"]
# print(train_set.shape, test_set.shape, valid_set.shape)
# lenslink_md.to_csv(os.path.join(md_datapath, 'processed/Lenselink_md.csv'))

# 筛选三大数据集
data_path = '/home/cpua212/code/ylyao/MDDTI_YYL/Data/'


# KIBA
kiba = DTI(name='KIBA', path=os.path.join(data_path, f'raw'))
# kiba.convert_to_log(form='binding')
kiba.harmonize_affinities('mean')
kiba = kiba.get_data()
kiba_md = kiba[kiba['Target'].isin(md_seq)]
kiba_md.to_csv(os.path.join(md_datapath, 'KIBA_md.csv'), sep=',', index=False)


# print('The information of kiba')
# print(kiba.head())
# print(kiba.shape)
# print(len(list(kiba['Drug_ID'].unique())))
# print(len(list(kiba['Target_ID'].unique())))
# kiba_md = kiba[kiba['Target'].isin(md_seq)]
# print('The information of kiba_md')
# print(len(list(kiba_md['Drug_ID'].unique())))
# print(len(list(kiba_md['Target_ID'].unique())))
# print(kiba_md.head())
# print(kiba_md.shape)
# kiba_md.to_csv(os.path.join(md_datapath, 'KIBA_md.csv'), sep=',', index=False)

# bindingdb_kd = DTI(name='BindingDB_Kd', path=os.path.join(data_path, f'raw'))
# bindingdb_kd.convert_to_log(form='binding')
# bindingdb_kd.harmonize_affinities('mean')
# bindingdb_kd = bindingdb_kd.get_data()
# print('The information of bindingdb_kd')
# print(bindingdb_kd.head())
# print(bindingdb_kd.shape)
# print(len(list(bindingdb_kd['Drug_ID'].unique())))
# print(len(list(bindingdb_kd['Target_ID'].unique())))
# bindingdb_kd_md = bindingdb_kd[bindingdb_kd['Target'].isin(md_seq)]
# print(bindingdb_kd_md.head())
# print(bindingdb_kd_md.shape)
# print(len(list(bindingdb_kd_md['Drug_ID'].unique())))
# print(len(list(bindingdb_kd_md['Target_ID'].unique())))
# bindingdb_kd_md.to_csv(os.path.join(md_datapath, 'BindingDBKd_md.csv'), sep=',', index=False)

# BindingDB_IC50
bindingdb_ic50 = DTI(name='BindingDB_IC50', path=os.path.join(data_path, f'raw'))
bindingdb_ic50.convert_to_log(form='binding')
bindingdb_ic50.harmonize_affinities('mean')
bindingdb_ic50 = bindingdb_ic50.get_data()
bindingdb_ic50_md = bindingdb_ic50[bindingdb_ic50['Target'].isin(md_seq)]
bindingdb_ic50_md.to_csv(os.path.join(md_datapath, 'BindingDBIC50_md.csv'), sep=',', index=False)


# print('The information of bindingdb_ic50')
# print(bindingdb_ic50.head())
# print(bindingdb_ic50.shape)
# print(len(list(bindingdb_ic50['Drug_ID'].unique())))
# print(len(list(bindingdb_ic50['Target_ID'].unique())))
# bindingdb_ic50_md = bindingdb_ic50[bindingdb_ic50['Target'].isin(md_seq)]
# print('The refined information of bindingdb_ic50')
# print(bindingdb_ic50_md.head())
# print(bindingdb_ic50_md.shape)
# print(len(list(bindingdb_ic50_md['Drug_ID'].unique())))
# print(len(list(bindingdb_ic50_md['Target_ID'].unique())))
# bindingdb_ic50_md.to_csv(os.path.join(md_datapath, 'BindingDBIC50_md.csv'), sep=',', index=False)



# bindingdb_ki = DTI(name = 'BindingDB_Ki', path=os.path.join(data_path, f'raw'))
# bindingdb_ki.convert_to_log(form='binding')
# bindingdb_ki.harmonize_affinities('mean')
# bindingdb_ki = bindingdb_ki.get_data()
# print('The information of bindingdb_ki')
# print(bindingdb_ki.head())
# print(bindingdb_ki.shape)
# print(len(list(bindingdb_ki['Drug_ID'].unique())))
# print(len(list(bindingdb_ki['Target_ID'].unique())))
# bindingdb_ki_md = bindingdb_ki[bindingdb_ki['Target'].isin(md_seq)]
# print('The refined information of bindingdb_ki')
# print(bindingdb_ki_md.head())
# print(bindingdb_ki_md.shape)
# print(len(list(bindingdb_ki_md['Drug_ID'].unique())))
# print(len(list(bindingdb_ki_md['Target_ID'].unique())))
# bindingdb_ki_md.to_csv(os.path.join(md_datapath, 'BindingDBKi_md.csv'), sep=',', index=False)








