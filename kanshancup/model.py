#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 查看个人上传数据目录
# ls ../kernel/

# 查看个人工作区文件
# ls ../work/

# 查看当前挂载的比赛数据集目录
# ls ../data/

# 参考

# https://www.biendata.com/models/category/2897/L_notebook

# In[2]:


#!pip install lightgbm

# In[ ]:


# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import logging

log_fmt = "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
logging.basicConfig(format=log_fmt, level=logging.INFO)

import warnings
warnings.filterwarnings('ignore')


def extract_day(s):
    return s.apply(lambda x: int(x.split('-')[0][1:]))


def extract_hour(s):
    return s.apply(lambda x: int(x.split('-')[1][1:]))

def parse_list_1(d):
    if d == '-1':
        return [0]
    return list(map(lambda x: int(x[1:]), str(d).split(',')))

def parse_list_2(d):
    if d == '-1':
        return [0]
    return list(map(lambda x: int(x[2:]), str(d).split(',')))

#base_path = '/mnt/wz2/zhangx/zx'

# 加载邀请回答数据

train = pd.read_csv('invite_info_0926.txt', sep='\t', header=None)
train.columns = ['qid', 'uid', 'dt', 'label']
logging.info("invite %s", train.shape)

test = pd.read_csv('invite_info_evaluate_2_0926.txt', sep='\t', header=None)
test.columns = ['qid', 'uid', 'dt']
logging.info("test %s", test.shape)

sub = test.copy()

sub_size = len(sub)

train['day'] = extract_day(train['dt'])
train['hour'] = extract_hour(train['dt'])

test['day'] = extract_day(test['dt'])
test['hour'] = extract_hour(test['dt'])

del train['dt'], test['dt']

# 加载问题
ques = pd.read_csv('question_info_0926.txt', header=None, sep='\t')
ques.columns = ['qid', 'q_dt', 'title_t1', 'title_t2', 'desc_t1', 'desc_t2', 'topic']
ques['title_t1']=ques['title_t1'].apply(parse_list_2)
ques['title_t2']=ques['title_t2'].apply(parse_list_1)
ques['desc_t1']=ques['desc_t1'].apply(parse_list_2)
ques['desc_t2']=ques['desc_t2'].apply(parse_list_1)

ques['title_t1_length']=ques['title_t1'].apply(len)
ques['title_t2_length']=ques['title_t2'].apply(len)
ques['desc_t1_length']=ques['desc_t1'].apply(len)
ques['desc_t2_length']=ques['desc_t2'].apply(len)

del ques['title_t1'], ques['title_t2'], ques['desc_t1'], ques['desc_t2']
logging.info("ques %s", ques.shape)
#del ques_copy['q_dt'], ques_copy['title_t1_length'],ques_copy['title_t2_length'],ques_copy['desc_t1_length'],ques_copy['desc_t2_length']
ques['q_day'] = extract_day(ques['q_dt'])
ques['q_hour'] = extract_hour(ques['q_dt'])
del ques['q_dt']
ques_copy=ques.copy()
del ques_copy['q_hour']
del ques['title_t1_length'],ques['title_t2_length'],ques['desc_t1_length'],ques['desc_t2_length']


# 加载回答
ans = pd.read_csv('answer_info_0926.txt', header=None, sep='\t')
ans.columns = ['aid', 'qid', 'uid', 'ans_dt', 'ans_t1', 'ans_t2', 'is_good', 'is_rec', 'is_dest', 'has_img',
               'has_video', 'word_count', 'reci_cheer', 'reci_uncheer', 'reci_comment', 'reci_mark', 'reci_tks',
               'reci_xxx', 'reci_no_help', 'reci_dis']
del ans['ans_t1'], ans['ans_t2']
logging.info("ans %s", ans.shape)

ans['a_day'] = extract_day(ans['ans_dt'])
ans['a_hour'] = extract_hour(ans['ans_dt'])
del ans['ans_dt']

ans = pd.merge(ans, ques, on='qid')
del ques

# 回答距提问的天数
ans['diff_qa_days'] = ans['a_day'] - ans['q_day']

# 时间窗口划分
# train
# val
print('train_day_min')
print(train['day'].min())
print('ans_day_min')
print(ans['a_day'].min())
print('ans_day_max')
print(ans['a_day'].max())
train_start = 3838
train_end = 3867

val_start = 3868
val_end = 3874

label_end = 3867
label_start = label_end - 6#3861

train_label_feature_end = label_end - 7#3860
train_label_feature_start = train_label_feature_end - 22#3838

train_ans_feature_end = label_end - 7#3860
train_ans_feature_start = train_ans_feature_end - 50-3#3810

val_label_feature_end = val_start - 1#3867
val_label_feature_start = val_label_feature_end - 22-7#3845

val_ans_feature_end = val_start - 1#3867
val_ans_feature_start = val_ans_feature_end - 50-10#3817

train_label_feature = train[(train['day'] >= train_label_feature_start) & (train['day'] <= train_label_feature_end)]
logging.info("train_label_feature %s", train_label_feature.shape)#3838-3860

val_label_feature = train[(train['day'] >= val_label_feature_start) & (train['day'] <= val_label_feature_end)]
logging.info("val_label_feature %s", val_label_feature.shape)#3845-3867

train_label = train[(train['day'] > train_label_feature_end)]#3860-

logging.info("train feature start %s end %s, label start %s end %s", train_label_feature['day'].min(),
             train_label_feature['day'].max(), train_label['day'].min(), train_label['day'].max())

logging.info("test feature start %s end %s, label start %s end %s", val_label_feature['day'].min(),
             val_label_feature['day'].max(), test['day'].min(), test['day'].max())

# 确定ans的时间范围
# 3807~3874
train_ans_feature = ans[(ans['a_day'] >= train_ans_feature_start) & (ans['a_day'] <= train_ans_feature_end)]

val_ans_feature = ans[(ans['a_day'] >= val_ans_feature_start) & (ans['a_day'] <= val_ans_feature_end)]

logging.info("train ans feature %s, start %s end %s", train_ans_feature.shape, train_ans_feature['a_day'].min(),
             train_ans_feature['a_day'].max())

logging.info("val ans feature %s, start %s end %s", val_ans_feature.shape, val_ans_feature['a_day'].min(),
             val_ans_feature['a_day'].max())

fea_cols = ['is_good', 'is_rec', 'is_dest', 'has_img', 'has_video', 'word_count',
            'reci_cheer', 'reci_uncheer', 'reci_comment', 'reci_mark', 'reci_tks',
            'reci_xxx', 'reci_no_help', 'reci_dis', 'diff_qa_days']


def extract_feature1(target, label_feature, ans_feature):
    # 问题特征
    t1 = label_feature.groupby('qid')['label'].agg(['mean', 'sum', 'std', 'count']).reset_index()
    t1.columns = ['qid', 'q_inv_mean', 'q_inv_sum', 'q_inv_std', 'q_inv_count']
    target = pd.merge(target, t1, on='qid', how='left')

    # 用户特征
    t1 = label_feature.groupby('uid')['label'].agg(['mean', 'sum', 'std', 'count']).reset_index()
    t1.columns = ['uid', 'u_inv_mean', 'u_inv_sum', 'u_inv_std', 'u_inv_count']
    target = pd.merge(target, t1, on='uid', how='left')
    
    #
    # train_size = len(train)
    # data = pd.concat((train, test), sort=True)

    # 回答部分特征

    t1 = ans_feature.groupby('qid')['aid'].count().reset_index()
    t1.columns = ['qid', 'q_ans_count']
    target = pd.merge(target, t1, on='qid', how='left')

    t1 = ans_feature.groupby('uid')['aid'].count().reset_index()
    t1.columns = ['uid', 'u_ans_count']
    target = pd.merge(target, t1, on='uid', how='left')
    #target=pd.merge(target,ans_feature,on=['qid','uid'],how='left')
    for col in fea_cols:
        t1 = ans_feature.groupby('uid')[col].agg(['sum', 'max', 'mean']).reset_index()
        #t1.columns = ['uid', f'u_{col}_sum', f'u_{col}_max', f'u_{col}_mean']
        t1.columns = ['uid', 'u_'+col+'_sum', 'u_'+col+'_max', 'u_'+col+'_mean']
        target = pd.merge(target, t1, on='uid', how='left')

        t1 = ans_feature.groupby('qid')[col].agg(['sum', 'max', 'mean']).reset_index()
        #t1.columns = ['qid', f'q_{col}_sum', f'q_{col}_max', f'q_{col}_mean']
        t1.columns = ['qid', 'q_'+col+'_sum', 'q_'+col+'_max', 'q_'+col+'_mean']
        target = pd.merge(target, t1, on='qid', how='left')
        logging.info("extract %s", col)

    

    return target



train_label = extract_feature1(train_label, train_label_feature, train_ans_feature)
test = extract_feature1(test, val_label_feature, val_ans_feature)
train_label=pd.merge(train_label,ques_copy,on='qid',how='left')
test=pd.merge(test,ques_copy,on='qid',how='left')

# 特征提取结束
logging.info("train shape %s, test shape %s", train_label.shape, test.shape)
assert len(test) == sub_size

# 加载用户
user = pd.read_csv('member_info_0926.txt', header=None, sep='\t')
user.columns = ['uid', 'gender', 'creat_keyword', 'level', 'hot', 'reg_type', 'reg_plat', 'freq', 'uf_b1', 'uf_b2',
                'uf_b3', 'uf_b4', 'uf_b5', 'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5', 'score', 'follow_topic',
                'inter_topic']
#del user['follow_topic'], user['inter_topic']


logging.info("user %s", user.shape)

unq = user.nunique()
logging.info("user unq %s", unq)
for x in unq[unq == 1].index:
    del user[x]
    logging.info('del unq==1 %s', x)

t = user.dtypes
cats = [x for x in t[t == 'object'].index if x not in ['follow_topic', 'inter_topic', 'uid']]
logging.info("user cat %s", cats)

for d in cats:
    lb = LabelEncoder()
    user[d] = lb.fit_transform(user[d])
    logging.info('encode %s', d)

q_lb = LabelEncoder()
q_lb.fit(list(train_label['qid'].astype(str).values) + list(test['qid'].astype(str).values))
train_label['qid_enc'] = q_lb.transform(train_label['qid'])
test['qid_enc'] = q_lb.transform(test['qid'])

u_lb = LabelEncoder()
u_lb.fit(user['uid'])
train_label['uid_enc'] = u_lb.transform(train_label['uid'])
test['uid_enc'] = u_lb.transform(test['uid'])

# merge user
train_label = pd.merge(train_label, user, on='uid', how='left')
test = pd.merge(test, user, on='uid', how='left')
logging.info("train shape %s, test shape %s", train_label.shape, test.shape)


data = pd.concat((train_label, test), axis=0, sort=True)
data['y_q']=data['day']-data['q_day']
tmp=['title_t1_length','title_t2_length','desc_t1_length','desc_t2_length','y_q']
for i in tmp:
    tmp_lb=LabelEncoder()
    data[i]=tmp_lb.fit_transform(data[i])
    



'''
def parse_list_1(d):
    if d == '-1':
        return []
    return list(map(lambda x: int(x[1:]), str(d).split(',')))

def parse_list_2(d):
    if d == '-1':
        return []
    return list(map(lambda x: int(x[1:]), str(d).split(',')))'''
def parse_map(d):
    if d == '-1':
        return {}
    return dict([int(z.split(':')[0][1:]), float(z.split(':')[1])] for z in d.split(','))


data['follow_topic'] = data['follow_topic'].apply(parse_list_1)
data['inter_topic'] = data['inter_topic'].apply(parse_map)
def get_interest_values(d):
    if len(d) == 0:
        return [0]
    return list(d.values())

# 用户topic兴趣值的统计特征
data['interest_values'] = data['inter_topic'].apply(get_interest_values)
data['min_interest_values'] = data['interest_values'].apply(np.min)
data['max_interest_values'] = data['interest_values'].apply(np.max)
data['mean_interest_values'] = data['interest_values'].apply(np.mean)
data['std_interest_values'] = data['interest_values'].apply(np.std)


data['topic']=data['topic'].apply(parse_list_1)
data['fl_tp']=data.apply(lambda row: list(set(row['topic'])&set(row['follow_topic'])),axis=1)
data['in_tp']=data.apply(lambda row: list(set(row['topic'])&set(row['inter_topic'].keys())),axis=1)

data['num_fl_tp']=data['fl_tp'].apply(len)
data['num_in_tp']=data['in_tp'].apply(len)
data['topic_in_tp']=data.apply(lambda row: [row['inter_topic'][t] for t in row['in_tp']],axis=1)
data['topic_interest_intersection_values'] = data['topic_in_tp'].apply(lambda x: [0] if len(x) == 0 else x)
data['min_topic_interest_intersection_values'] = data['topic_interest_intersection_values'].apply(np.min)
data['max_topic_interest_intersection_values'] = data['topic_interest_intersection_values'].apply(np.max)
data['mean_topic_interest_intersection_values'] = data['topic_interest_intersection_values'].apply(np.mean)
data['std_topic_interest_intersection_values'] = data['topic_interest_intersection_values'].apply(np.std)

data['topic_count']=data.apply(lambda row: len(set(row['topic'])),axis=1)
data['follow_topic_count']=data.apply(lambda row: len(set(row['follow_topic'])),axis=1)
data['inter_topic_count']=data.apply(lambda row: len(set(row['inter_topic'])),axis=1)
def most_interest_topic(d):
    if len(d) == 0:
        return -1
    return list(d.keys())[np.argmax(list(d.values()))]

# 用户最感兴趣的topic
data['most_interest_topic'] = data['inter_topic'].apply(most_interest_topic)
data['most_interest_topic'] = LabelEncoder().fit_transform(data['most_interest_topic'])

del data['topic']
del data['follow_topic'],data['topic_in_tp'],data['interest_values']
del data['inter_topic'],data['fl_tp'],data['in_tp'],data['topic_interest_intersection_values']
#del train_label, test
# count编码
count_fea = ['uid_enc', 'qid_enc', 'gender', 'freq', 'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5']
for feat in count_fea:
    col_name = '{}_count'.format(feat)
    data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
    data.loc[data[col_name] < 2, feat] = -1
    data[feat] += 1
    data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
    data[col_name] = (data[col_name] - data[col_name].min()) / (data[col_name].max() - data[col_name].min())
    # 

# 问题被回答的次数


# 压缩数据
t = data.dtypes
for x in t[t == 'int64'].index:
    data[x] = data[x].astype('int32')

for x in t[t == 'float64'].index:
    data[x] = data[x].astype('float32')

data['wk'] = data['day'] % 7

feature_cols = [x for x in data.columns if x not in ('label', 'uid', 'qid', 'dt', 'day','q_day')]
#feature_cols = [x for x in data.columns if x not in ('label', 'uid', 'qid', 'dt', 'day')]

# target编码
logging.info("feature size %s", len(feature_cols))

X_train_all = data.iloc[:len(train_label)][feature_cols]
y_train_all = data.iloc[:len(train_label)]['label']
test = data.iloc[len(train_label):]
del data
assert len(test) == sub_size

logging.info("train shape %s, test shape %s", train_label.shape, test.shape)

fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for index, (train_idx, val_idx) in enumerate(fold.split(X=X_train_all, y=y_train_all)):
    break

X_train, X_val, y_train, y_val = X_train_all.iloc[train_idx][feature_cols], X_train_all.iloc[val_idx][feature_cols], \
                                 y_train_all.iloc[train_idx], \
                                 y_train_all.iloc[val_idx]
del X_train_all

model_lgb = LGBMClassifier(n_estimators=2000, n_jobs=-1, objective='binary', seed=1000, silent=True)
model_lgb.fit(X_train, y_train,
              eval_metric=['logloss', 'auc'],
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=50)

#lgb_feat_imp = pd.Series(model_lgb.feature_importances_, feature_cols).sort_values(ascending=False)
'''lgb_feat_imp=pd.DataFrame({'column': feature_cols,'importance': model_lgb.feature_importance(),
    }).sort_values(by='importance')

lgb_feat_imp.to_csv('lgb_feat_imp.csv')'''
booster = model_lgb.booster_
importance = booster.feature_importance(importance_type='split')
feature_name = booster.feature_name()
# for (feature_name,importance) in zip(feature_name,importance):
#     print (feature_name,importance) 
feature_importance = pd.DataFrame({'feature_name':feature_name,'importance':importance} )
feature_importance.to_csv('feature_importance.csv',index=False)

sub['label'] = model_lgb.predict_proba(test[feature_cols])[:, 1]
sub.to_csv('result11.txt', index=None, header=None, sep='\t')


# In[ ]:



