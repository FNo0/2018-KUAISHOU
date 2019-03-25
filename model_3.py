# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 00:36:17 2018

@author: FNo0
"""

import pandas as pd
import numpy as np
import pickle
import re
from functools import reduce 
import datetime
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

def load_data():
    # 源数据a
    a_app_launch_log = pd.read_table('data/a3d6_chusai_a_train/app_launch_log.txt',\
                                   delimiter = '	',\
                                   names = ['user_id','day'])
    a_user_activity_log = pd.read_table('data/a3d6_chusai_a_train/user_activity_log.txt',\
                                     delimiter = '	',\
                                     names = ['user_id','day','page','video_id','author_id','action_type'])
    a_user_register_log = pd.read_table('data/a3d6_chusai_a_train/user_register_log.txt',\
                                     delimiter = '	',\
                                     names = ['user_id','register_day','register_type','device_type'])
    a_video_create_log = pd.read_table('data/a3d6_chusai_a_train/video_create_log.txt',\
                                     delimiter = '	',\
                                     names = ['user_id','day'])
    # 源数据b
    b_app_launch_log = pd.read_table('data/chusai_b_train/app_launch_log.txt',\
                                   delimiter = '	',\
                                   names = ['user_id','day'])
    b_user_activity_log = pd.read_table('data/chusai_b_train/user_activity_log.txt',\
                                      delimiter = '	',\
                                      names = ['user_id','day','page','video_id','author_id','action_type'])
    b_user_register_log = pd.read_table('data/chusai_b_train/user_register_log.txt',\
                                      delimiter = '	',\
                                      names = ['user_id','register_day','register_type','device_type'])
    b_video_create_log = pd.read_table('data/chusai_b_train/video_create_log.txt',\
                                     delimiter = '	',\
                                     names = ['user_id','day'])
    # 重编码a的user_id
    a_app_launch_log['user_id'] = a_app_launch_log['user_id'].map(lambda x : 'a_' + str(x))
    a_user_activity_log['user_id'] = a_user_activity_log['user_id'].map(lambda x : 'a_' + str(x))
    a_user_register_log['user_id'] = a_user_register_log['user_id'].map(lambda x : 'a_' + str(x))
    a_video_create_log['user_id'] = a_video_create_log['user_id'].map(lambda x : 'a_' + str(x))
    # 重编码a的author_id
    a_user_activity_log['author_id'] = a_user_activity_log['author_id'].map(lambda x : 'a_' + str(x))
    # 重编码a的video_id
    a_user_activity_log['video_id'] = a_user_activity_log['video_id'].map(lambda x : 'a_' + str(x))
    # 重编码b的user_id
    b_app_launch_log['user_id'] = b_app_launch_log['user_id'].map(lambda x : 'b_' + str(x))
    b_user_activity_log['user_id'] = b_user_activity_log['user_id'].map(lambda x : 'b_' + str(x))
    b_user_register_log['user_id'] = b_user_register_log['user_id'].map(lambda x : 'b_' + str(x))
    b_video_create_log['user_id'] = b_video_create_log['user_id'].map(lambda x : 'b_' + str(x))
    # 重编码b的author_id
    b_user_activity_log['author_id'] = b_user_activity_log['author_id'].map(lambda x : 'b_' + str(x))
    # 重编码b的video_id
    b_user_activity_log['video_id'] = b_user_activity_log['video_id'].map(lambda x : 'b_' + str(x))
    # 合并
    app_launch_log = pd.concat([b_app_launch_log,a_app_launch_log],axis = 0)
    user_activity_log = pd.concat([b_user_activity_log,a_user_activity_log],axis = 0)
    user_register_log = pd.concat([b_user_register_log,a_user_register_log],axis = 0)
    video_create_log = pd.concat([b_video_create_log,a_video_create_log],axis = 0)
    # 重置index
    app_launch_log.index = range(len(app_launch_log))
    user_activity_log.index = range(len(user_activity_log))
    user_register_log.index = range(len(user_register_log))
    video_create_log.index = range(len(video_create_log))
    # 返回
    return app_launch_log,user_activity_log,user_register_log,video_create_log

def get_user(register,user_dates):
    # 注册日在user_dates[-1]前的所有用户
    user = register[register['register_day'].map(lambda x : x <= user_dates[-1])][['user_id']]
    # 返回
    return user

def get_label(launch,activity,video,label_dates):
    # 分别在launch、activity、video里面活跃的用户
    pos_in_launch = launch[launch['day'].map(lambda x : x in label_dates)].drop_duplicates(['user_id'],keep = 'first')
    pos_in_activity = activity[activity['day'].map(lambda x : x in label_dates)].drop_duplicates(['user_id'],keep = 'first')
    pos_in_video = video[video['day'].map(lambda x : x in label_dates)].drop_duplicates(['user_id'],keep = 'first')
    # 活跃的用户
    label = list(set(pos_in_launch['user_id'].tolist() + pos_in_activity['user_id'].tolist() + pos_in_video['user_id'].tolist()))
    label = pd.DataFrame(label,columns = ['user_id'])
    label['label'] = 1
    # 返回
    return label

def get_base_feat(register):
    # 返回的特征
    feature = register.drop(['register_day'],axis = 1)
    # 离散register_type
    df = pd.get_dummies(feature['register_type'],prefix = 'register_type')
    feature = pd.concat([feature,df],axis = 1)
    # 返回
    return feature

def get_register_feat(register,user_dates):
    # 源数据
    history = register[register['register_day'].map(lambda x : x in user_dates)]
    # 返回的特征
    feature = history[['user_id','register_day']]
    # 注册日据最近考察日间隔
    feature['label_sub_register'] = feature['register_day'].map(lambda x : user_dates[-1] + 1 - x)
    # 删不需要的
    feature.drop(['register_day'],axis = 1,inplace = True)
    # 返回
    return feature

def get_launch_feat(launch,feat_dates):
    # 源数据
    history = launch[launch['day'].map(lambda x : x in feat_dates)]
    history['cnt'] = 1
    # 返回的特征
    feature = pd.DataFrame(columns = ['user_id'])
    
    ## 统计特征
    pivot = pd.pivot_table(history,index = ['user_id','day'],values = 'cnt',aggfunc = len)
    pivot = pivot.unstack(level = -1)
    pivot.fillna(0,downcast = 'infer',inplace = True)
    feat = pd.DataFrame()
    feat['user_id'] = pivot.index
    feat.index = pivot.index
    # 每一天的特征
    for i in range(1,len(feat_dates) + 1):
        feat['user_launch_cnt_before_' + str(i) + '_day'] = pivot[pivot.columns.tolist()[-i]]
    # 总和
    feat['user_launch_cnt_sum'] = pivot.sum(1)
    # 均值
    feat['user_launch_cnt_mean'] = pivot.mean(1)
    # 方差
    feat['user_launch_cnt_var'] = pivot.var(1)
    # 最大值
    feat['user_launch_cnt_max'] = pivot.max(1)
    # 最小值
    feat['user_launch_cnt_min'] = pivot.min(1)
    # 加入feature
    feature = pd.merge(feature,feat,on = ['user_id'],how = 'outer')
    
#    ## 差分与统计
#    diff = pivot.diff(axis = 1)
#    diff = diff[diff.columns.tolist()[1:]]
#    feat = pd.DataFrame()
#    feat['user_id'] = diff.index
#    feat.index = diff.index
#    # 每一个差分
#    for i in range(1,len(feat_dates)):
#        feat['user_launch_diff_before_' + str(i) + '_day'] = diff[diff.columns.tolist()[-i]]
#    # 总和
#    feat['user_launch_diff_sum'] = diff.sum(1)
#    # 均值
#    feat['user_launch_diff_mean'] = diff.mean(1)
#    # 方差
#    feat['user_launch_diff_var'] = diff.var(1)
#    # 最大值
#    feat['user_launch_diff_max'] = diff.max(1)
#    # 最小值
#    feat['user_launch_diff_min'] = diff.min(1)
#    # 加入feature
#    feature = pd.merge(feature,feat,on = ['user_id'],how = 'outer')
    
    ## 连续登陆
    feat = pd.DataFrame()
    feat['user_id'] = pivot.index
    feat.index = pivot.index
    pivot = pivot.applymap(lambda x : 1 if x != 0 else 0)
    feat['launch_list'] = pivot.apply(lambda x : reduce(lambda y,z : str(y) + str(z),x),axis = 1)
    # 连续登陆天数_均值
    feat['user_launch_continue_mean'] = feat['launch_list'].map(lambda x : np.mean([len(y) for y in re.split('0+',x.strip('0'))]))
    # 连续登陆天数_方差
    feat['user_launch_continue_var'] = feat['launch_list'].map(lambda x : np.var([len(y) for y in re.split('0+',x.strip('0'))]))
    # 连续登陆天数_最大值
    feat['user_launch_continue_max'] = feat['launch_list'].map(lambda x : np.max([len(y) for y in re.split('0+',x.strip('0'))]))
    # 连续登陆天数_最小值
    feat['user_launch_continue_min'] = feat['launch_list'].map(lambda x : np.min([len(y) for y in re.split('0+',x.strip('0'))]))
    # 去掉无用的
    feat.drop(['launch_list'],axis = 1,inplace = True)
    # 加入feature
    feature = pd.merge(feature,feat,on = ['user_id'],how = 'outer')
    
    ## 时间间隔
    # 最近/远一次启动距离最近考察日的时间间隔
    near = 'nearest_day_launch'
    fur = 'furest_day_launch'
    pivot_n = pd.pivot_table(history,index = ['user_id'],values = 'day',aggfunc = max)
    pivot_n.rename(columns = {'day' : near},inplace = True)
    pivot_n.reset_index(inplace = True)
    pivot_f = pd.pivot_table(history,index = ['user_id'],values = 'day',aggfunc = min)
    pivot_f.rename(columns = {'day' : fur},inplace = True)
    pivot_f.reset_index(inplace = True)
    feature = pd.merge(feature,pivot_n,on = ['user_id'],how = 'left')
    feature = pd.merge(feature,pivot_f,on = ['user_id'],how = 'left')
    feature[near + '_to_label'] = feature[near].map(lambda x : feat_dates[-1] + 1 - x)
    feature[fur + '_to_label'] = feature[fur].map(lambda x : feat_dates[-1] + 1 - x)
    feature.drop([near,fur],axis = 1,inplace = True)
    
    ## 填空
    feature.fillna(0,downcast = 'infer',inplace = True)
    ## 返回
    return feature

def get_activity_feat(activity,feat_dates):
    # 源数据
    history = activity[activity['day'].map(lambda x : x in feat_dates)]
    history['cnt'] = 1
    # 返回的特征
    feature = pd.DataFrame(columns = ['user_id'])
    
    ## 统计特征
    pivot = pd.pivot_table(history,index = ['user_id','day'],values = 'cnt',aggfunc = len)
    pivot = pivot.unstack(level = -1)
    pivot.fillna(0,downcast = 'infer',inplace = True)
    feat = pd.DataFrame()
    feat['user_id'] = pivot.index
    feat.index = pivot.index
    # 每一天的特征
    for i in range(1,len(feat_dates) + 1):
        feat['user_activity_cnt_before_' + str(i) + '_day'] = pivot[pivot.columns.tolist()[-i]]
    # 总和
    feat['user_activity_cnt_sum'] = pivot.sum(1)
    # 均值
    feat['user_activity_cnt_mean'] = pivot.mean(1)
    # 方差
    feat['user_activity_cnt_var'] = pivot.var(1)
    # 最大值
    feat['user_activity_cnt_max'] = pivot.max(1)
    # 最小值
    feat['user_activity_cnt_min'] = pivot.min(1)
    # 加入feature
    feature = pd.merge(feature,feat,on = ['user_id'],how = 'outer')
    
#    ## 差分与统计
#    diff = pivot.diff(axis = 1)
#    diff = diff[diff.columns.tolist()[1:]]
#    feat = pd.DataFrame()
#    feat['user_id'] = diff.index
#    feat.index = diff.index
#    # 每一个差分
#    for i in range(1,len(feat_dates)):
#        feat['user_activity_diff_before_' + str(i) + '_day'] = diff[diff.columns.tolist()[-i]]
#    # 总和
#    feat['user_activity_diff_sum'] = diff.sum(1)
#    # 均值
#    feat['user_activity_diff_mean'] = diff.mean(1)
#    # 方差
#    feat['user_activity_diff_var'] = diff.var(1)
#    # 最大值
#    feat['user_activity_diff_max'] = diff.max(1)
#    # 最小值
#    feat['user_activity_diff_min'] = diff.min(1)
#    # 加入feature
#    feature = pd.merge(feature,feat,on = ['user_id'],how = 'outer')
    
    ## page、action_type特征
    # (关注页page = 0,个人主页page = 1,发现页page = 2,同城页page = 3,其他页page = 4)
    # (播放action_type = 0,关注action_type = 1,点赞action_type = 2,转发action_type = 3,举报action_type = 4,减少此类作品action_type = 5)
    for i in [0,1,2,3,4]:
        pivot = pd.pivot_table(history[history['page'] == i],index = ['user_id','day'],values = 'cnt',aggfunc = len)
        pivot = pivot.unstack(level = -1)
        pivot.fillna(0,downcast = 'infer',inplace = True)
        feat = pd.DataFrame()
        feat['user_id'] = pivot.index
        feat.index = pivot.index
        feat['user_page_' + str(i) + '_cnt_sum'] = pivot.sum(1)
        feat['user_page_' + str(i) + '_cnt_mean'] = pivot.mean(1)
        feat['user_page_' + str(i) + '_cnt_var'] = pivot.var(1)
        feat['user_page_' + str(i) + '_cnt_max'] = pivot.max(1)
        feat['user_page_' + str(i) + '_cnt_min'] = pivot.min(1)
        feature = pd.merge(feature,feat,on = ['user_id'],how = 'outer')
    for i in [0,1,2,3,4,5]:
        pivot = pd.pivot_table(history[history['action_type'] == i],index = ['user_id','day'],values = 'cnt',aggfunc = len)
        pivot = pivot.unstack(level = -1)
        pivot.fillna(0,downcast = 'infer',inplace = True)
        feat = pd.DataFrame()
        feat['user_id'] = pivot.index
        feat.index = pivot.index
        feat['user_action_type_' + str(i) + '_cnt_sum'] = pivot.sum(1)
        feat['user_action_type_' + str(i) + '_cnt_mean'] = pivot.mean(1)
        feat['user_action_type_' + str(i) + '_cnt_var'] = pivot.var(1)
        feat['user_action_type_' + str(i) + '_cnt_max'] = pivot.max(1)
        feat['user_action_type_' + str(i) + '_cnt_min'] = pivot.min(1)
        feature = pd.merge(feature,feat,on = ['user_id'],how = 'outer')
        
    ## 时间间隔
    # 最近/远一次活动距离最近考察日的时间间隔
    near = 'nearest_day_activity'
    fur = 'furest_day_activity'
    pivot_n = pd.pivot_table(history,index = ['user_id'],values = 'day',aggfunc = max)
    pivot_n.rename(columns = {'day' : near},inplace = True)
    pivot_n.reset_index(inplace = True)
    pivot_f = pd.pivot_table(history,index = ['user_id'],values = 'day',aggfunc = min)
    pivot_f.rename(columns = {'day' : fur},inplace = True)
    pivot_f.reset_index(inplace = True)
    feature = pd.merge(feature,pivot_n,on = ['user_id'],how = 'left')
    feature = pd.merge(feature,pivot_f,on = ['user_id'],how = 'left')
    feature[near + '_to_label'] = feature[near].map(lambda x : feat_dates[-1] + 1 - x)
    feature[fur + '_to_label'] = feature[fur].map(lambda x : feat_dates[-1] + 1 - x)
    feature.drop([near,fur],axis = 1,inplace = True)
    # 最近/远一次page活动距离最近考察日的时间间隔
    for i in [0,1,2,3,4]:
        near = 'nearest_day_page_' + str(i)
        fur = 'furest_day_page_' + str(i)
        pivot_n = pd.pivot_table(history[history['page'] == i],index = ['user_id'],values = 'day',aggfunc = max)
        pivot_n.rename(columns = {'day' : near},inplace = True)
        pivot_n.reset_index(inplace = True)
        pivot_f = pd.pivot_table(history[history['page'] == i],index = ['user_id'],values = 'day',aggfunc = min)
        pivot_f.rename(columns = {'day' : fur},inplace = True)
        pivot_f.reset_index(inplace = True)
        feature = pd.merge(feature,pivot_n,on = ['user_id'],how = 'left')
        feature = pd.merge(feature,pivot_f,on = ['user_id'],how = 'left')
        feature[near + '_to_label'] = feature[near].map(lambda x : feat_dates[-1] + 1 - x)
        feature[fur + '_to_label'] = feature[fur].map(lambda x : feat_dates[-1] + 1 - x)
        feature.drop([near,fur],axis = 1,inplace = True)
    # 最近/远一次action_type活动距离最近考察日的时间间隔
    for i in [0,1,2,3,4,5]:
        near = 'nearest_day_action_type_' + str(i)
        fur = 'furest_day_action_type_' + str(i)
        pivot_n = pd.pivot_table(history[history['action_type'] == i],index = ['user_id'],values = 'day',aggfunc = max)
        pivot_n.rename(columns = {'day' : near},inplace = True)
        pivot_n.reset_index(inplace = True)
        pivot_f = pd.pivot_table(history[history['action_type'] == i],index = ['user_id'],values = 'day',aggfunc = min)
        pivot_f.rename(columns = {'day' : fur},inplace = True)
        pivot_f.reset_index(inplace = True)
        feature = pd.merge(feature,pivot_n,on = ['user_id'],how = 'left')
        feature = pd.merge(feature,pivot_f,on = ['user_id'],how = 'left')
        feature[near + '_to_label'] = feature[near].map(lambda x : feat_dates[-1] + 1 - x)
        feature[fur + '_to_label'] = feature[fur].map(lambda x : feat_dates[-1] + 1 - x)
        feature.drop([near,fur],axis = 1,inplace = True)
        
    ## 关联度
    # 用户发布的视频总浏览数
    authors = pd.pivot_table(history,index = ['author_id'],values = 'user_id',aggfunc = len)
    authors.rename(columns = {'user_id' : 'author_cnt'},inplace = True)
    authors.reset_index(inplace = True)
    # 用户每个视频被浏览数
    videos = pd.pivot_table(history,index = ['author_id','video_id'],values = 'user_id',aggfunc = len)
    videos.rename(columns = {'user_id' : 'video_cnt'},inplace = True)
    videos.reset_index(inplace = True)
    # 合
    authors_videos = pd.merge(authors,videos,on = 'author_id',how = 'right')
    authors_videos['rate'] = authors_videos['video_cnt'] / authors_videos['author_cnt']
    authors_videos = authors_videos[['author_id','video_id','rate']]
    # 用户活动次数
    users = pd.pivot_table(history,index = ['user_id','video_id'],values = 'cnt',aggfunc = len)
    users.rename(columns = {'cnt' : 'user_cnt'},inplace = True)
    users.reset_index(inplace = True)
    # 合
    users_authors_videos = pd.merge(users,authors_videos,on = ['video_id'],how = 'left')
    # user-video关联度
    users_authors_videos['similar'] = users_authors_videos['rate'] * users_authors_videos['user_cnt']
    # user-author关联度
    users_authors_videos = pd.pivot_table(users_authors_videos,index = ['user_id','author_id'],values = 'similar',aggfunc = sum)
    users_authors_videos.reset_index(inplace = True)
    # 均值
    mean = pd.pivot_table(users_authors_videos,index = ['user_id'],values = 'similar',aggfunc = np.mean)
    mean.rename(columns = {'similar' : 'similar_mean'},inplace = True)
    mean.reset_index(inplace = True)
    # 方差
    var = pd.pivot_table(users_authors_videos,index = ['user_id'],values = 'similar',aggfunc = np.var)
    var.rename(columns = {'similar' : 'similar_var'},inplace = True)
    var.reset_index(inplace = True)
    # 最大值
    maxs = pd.pivot_table(users_authors_videos,index = ['user_id'],values = 'similar',aggfunc = max)
    maxs.rename(columns = {'similar' : 'similar_max'},inplace = True)
    maxs.reset_index(inplace = True)
    # 最小值
    mins = pd.pivot_table(users_authors_videos,index = ['user_id'],values = 'similar',aggfunc = min)
    mins.rename(columns = {'similar' : 'similar_min'},inplace = True)
    mins.reset_index(inplace = True)
    ## 合并
    feature = pd.merge(feature,mean,on = 'user_id',how = 'left')
    feature = pd.merge(feature,var,on = 'user_id',how = 'left')
    feature = pd.merge(feature,maxs,on = 'user_id',how = 'left')
    feature = pd.merge(feature,mins,on = 'user_id',how = 'left')
        
    ## 填空
    feature.fillna(0,downcast = 'infer',inplace = True)
    ## 返回
    return feature

def get_video_feat(video,feat_dates):
    # 源数据
    history = video[video['day'].map(lambda x : x in feat_dates)]
    history['cnt'] = 1
    # 返回的特征
    feature = pd.DataFrame(columns = ['user_id'])
    
    ## 统计特征
    pivot = pd.pivot_table(history,index = ['user_id','day'],values = 'cnt',aggfunc = len)
    pivot = pivot.unstack(level = -1)
    pivot.fillna(0,downcast = 'infer',inplace = True)
    feat = pd.DataFrame()
    feat['user_id'] = pivot.index
    feat.index = pivot.index
    # 每一天的特征
    for i in range(1,len(feat_dates) + 1):
        feat['user_video_cnt_before_' + str(i) + '_day'] = pivot[pivot.columns.tolist()[-i]]
    # 总和
    feat['user_video_cnt_sum'] = pivot.sum(1)
    # 均值
    feat['user_video_cnt_mean'] = pivot.mean(1)
    # 方差
    feat['user_video_cnt_var'] = pivot.var(1)
    # 最大值
    feat['user_video_cnt_max'] = pivot.max(1)
    # 最小值
    feat['user_video_cnt_min'] = pivot.min(1)
    # 加入feature
    feature = pd.merge(feature,feat,on = ['user_id'],how = 'outer')
    
#    ## 差分与统计
#    diff = pivot.diff(axis = 1)
#    diff = diff[diff.columns.tolist()[1:]]
#    feat = pd.DataFrame()
#    feat['user_id'] = diff.index
#    feat.index = diff.index
#    # 每一个差分
#    for i in range(1,len(feat_dates)):
#        feat['user_video_diff_before_' + str(i) + '_day'] = diff[diff.columns.tolist()[-i]]
#    # 总和
#    feat['user_video_diff_sum'] = diff.sum(1)
#    # 均值
#    feat['user_video_diff_mean'] = diff.mean(1)
#    # 方差
#    feat['user_video_diff_var'] = diff.var(1)
#    # 最大值
#    feat['user_video_diff_max'] = diff.max(1)
#    # 最小值
#    feat['user_video_diff_min'] = diff.min(1)
#    # 加入feature
#    feature = pd.merge(feature,feat,on = ['user_id'],how = 'outer')
    
    ## 连续拍摄
    feat = pd.DataFrame()
    feat['user_id'] = pivot.index
    feat.index = pivot.index
    pivot = pivot.applymap(lambda x : 1 if x != 0 else 0)
    feat['video_list'] = pivot.apply(lambda x : reduce(lambda y,z : str(y) + str(z),x),axis = 1)
    # 连续拍摄天数_均值
    feat['user_video_continue_mean'] = feat['video_list'].map(lambda x : np.mean([len(y) for y in re.split('0+',x.strip('0'))]))
    # 连续拍摄天数_方差
    feat['user_video_continue_var'] = feat['video_list'].map(lambda x : np.var([len(y) for y in re.split('0+',x.strip('0'))]))
    # 连续拍摄天数_最大值
    feat['user_video_continue_max'] = feat['video_list'].map(lambda x : np.max([len(y) for y in re.split('0+',x.strip('0'))]))
    # 连续拍摄天数_最小值
    feat['user_video_continue_min'] = feat['video_list'].map(lambda x : np.min([len(y) for y in re.split('0+',x.strip('0'))]))
    # 去掉无用的
    feat.drop(['video_list'],axis = 1,inplace = True)
    # 加入feature
    feature = pd.merge(feature,feat,on = ['user_id'],how = 'outer')
    
    ## 时间间隔
    # 最近/远一次拍摄距离最近考察日的时间间隔
    near = 'nearest_day_video'
    fur = 'furest_day_video'
    pivot_n = pd.pivot_table(history,index = ['user_id'],values = 'day',aggfunc = max)
    pivot_n.rename(columns = {'day' : near},inplace = True)
    pivot_n.reset_index(inplace = True)
    pivot_f = pd.pivot_table(history,index = ['user_id'],values = 'day',aggfunc = min)
    pivot_f.rename(columns = {'day' : fur},inplace = True)
    pivot_f.reset_index(inplace = True)
    feature = pd.merge(feature,pivot_n,on = ['user_id'],how = 'left')
    feature = pd.merge(feature,pivot_f,on = ['user_id'],how = 'left')
    feature[near + '_to_label'] = feature[near].map(lambda x : feat_dates[-1] + 1 - x)
    feature[fur + '_to_label'] = feature[fur].map(lambda x : feat_dates[-1] + 1 - x)
    feature.drop([near,fur],axis = 1,inplace = True)
    
    ## 填空
    feature.fillna(0,downcast = 'infer',inplace = True)
    ## 返回
    return feature

def create_dataset(user,base_feat,register_feat,launch_feat,activity_feat,video_feat):
    # user为标准,左连base_feat
    data = pd.merge(user,base_feat,on = 'user_id',how = 'left')
    # 左连regisetr_feat
    data = pd.merge(data,register_feat,on = 'user_id',how = 'left')
    # 外连launch_feat,左连也行
    data = pd.merge(data,launch_feat,on = 'user_id',how = 'outer')
    # 外连activity_feat,左连也行
    data = pd.merge(data,activity_feat,on = 'user_id',how = 'outer')
    # 外连video_feat,左连也行
    data = pd.merge(data,video_feat,on = 'user_id',how = 'outer')
    # 填空
    data.fillna(0,downcast = 'infer',inplace = True)
    # 返回
    return data

def get_dataset(app_launch_log,user_activity_log,user_register_log,video_create_log):
    # 与时序无关的特征
    base_feat = get_base_feat(user_register_log)
    ## off_tr
    off_tr = pd.DataFrame()
    i = 0
    print(str(round(i/4*100)) + '%...',end = '')
    i += 1
    for start in [17,10]:
        # 获得该start前注册的所有用户
        user_dates = list(range(1,start))
        user = get_user(user_register_log,user_dates)
        # 打标
        label_dates = list(range(start,start + 7))
        label = get_label(app_launch_log,user_activity_log,video_create_log,label_dates)
        # 提特征
        feat_dates = list(range(start - 9,start))
        register_feat = get_register_feat(user_register_log,user_dates)
        launch_feat = get_launch_feat(app_launch_log,feat_dates)
        activity_feat = get_activity_feat(user_activity_log,feat_dates)
        video_feat = get_video_feat(video_create_log,feat_dates)
        # 构造集
        data = create_dataset(user,base_feat,register_feat,launch_feat,activity_feat,video_feat)
        # 加标签
        data = pd.merge(data,label,on = 'user_id',how = 'left')
        data.fillna(0,downcast = 'infer',inplace = True)
        print(str(round(i/4*100)) + '%...',end = '')
        i += 1
        # 累加训练集
        off_tr = pd.concat([off_tr,data],axis = 0)
        
    ## va
    start = 24
    user_dates = list(range(1,start))
    user = get_user(user_register_log,user_dates)
    # 打标
    label_dates = list(range(start,start + 7))
    label = get_label(app_launch_log,user_activity_log,video_create_log,label_dates)
    # 提特征
    feat_dates = list(range(start - 9,start))
    register_feat = get_register_feat(user_register_log,user_dates)
    launch_feat = get_launch_feat(app_launch_log,feat_dates)
    activity_feat = get_activity_feat(user_activity_log,feat_dates)
    video_feat = get_video_feat(video_create_log,feat_dates)
    # 构造集
    va = create_dataset(user,base_feat,register_feat,launch_feat,activity_feat,video_feat)
    # 加标签
    va = pd.merge(va,label,on = 'user_id',how = 'left')
    va.fillna(0,downcast = 'infer',inplace = True)
    print(str(round(i/4*100)) + '%...',end = '')
    i += 1
    
    ## on_tr
    on_tr = pd.concat([va,off_tr],axis = 0)
    
    ## te
    start = 31
    user_dates = list(range(1,start))
    user = get_user(user_register_log,user_dates)
    # 提特征
    feat_dates = list(range(start - 9,start))
    register_feat = get_register_feat(user_register_log,user_dates)
    launch_feat = get_launch_feat(app_launch_log,feat_dates)
    activity_feat = get_activity_feat(user_activity_log,feat_dates)
    video_feat = get_video_feat(video_create_log,feat_dates)
    # 构造集
    te = create_dataset(user,base_feat,register_feat,launch_feat,activity_feat,video_feat)
    print(str(round(i/4*100)) + '%')
    i += 1
    
    ## 
    va = va[va['user_id'].map(lambda x : 'b_' in x)]
    te = te[te['user_id'].map(lambda x : 'b_' in x)]
    va['user_id'] = va['user_id'].map(lambda x : int(x.split('_')[1]))
    te['user_id'] = te['user_id'].map(lambda x : int(x.split('_')[1]))
    
#    ## 保存
#    pickle.dump(off_tr,open('tmp/off_train_3.pkl','wb'),protocol = 4)
#    pickle.dump(va,open('tmp/off_validate_3.pkl','wb'),protocol = 4)
#    pickle.dump(on_tr,open('tmp/on_train_3.pkl','wb'),protocol = 4)
#    pickle.dump(te,open('tmp/on_test_3.pkl','wb'),protocol = 4)
    
    # 返回
    return off_tr,va,on_tr,te
    
def xgb_for_va(off_tr,va):
    train = off_tr.copy()
    validate = va.copy()
    
    train_y = train['label'].values
    train_x = train.drop(['user_id','label'],axis=1).values
    validate_x = validate.drop(['user_id','label'],axis=1).values
 
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalidate = xgb.DMatrix(validate_x)
    
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'binary:logistic',
              'eval_metric' : 'error',
              'eta': 0.03,
              'max_depth': 6,  # 4 3
              'colsample_bytree': 0.8,#0.8
              'subsample': 0.8,
              'scale_pos_weight': 1,
              'min_child_weight': 14  # 2 3
              }
    # 训练
    bst = xgb.train(params, dtrain, num_boost_round=240)
    # 预测
    predict = bst.predict(dvalidate)
    validate_xy = validate[['user_id']]
    validate_xy['predicted_score'] = predict
    validate_xy.sort_values(['predicted_score'],ascending = False,inplace = True)
    # 返回
    return validate_xy  

def xgb_for_te(on_tr,te):
    train = on_tr.copy()
    test = te.copy()
    
    train_y = train['label'].values
    train_x = train.drop(['user_id','label'],axis=1).values
    test_x = test.drop(['user_id'],axis=1).values
 
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)
    
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'binary:logistic',
              'eval_metric' : 'error',
              'eta': 0.03,
              'max_depth': 6,  # 4 3
              'colsample_bytree': 0.8,#0.8
              'subsample': 0.8,
              'scale_pos_weight': 1,
              'min_child_weight': 14  # 2 3
              }
    # 训练
    bst = xgb.train(params, dtrain, num_boost_round=240)
    # 预测
    predict = bst.predict(dtest)
    test_xy = test[['user_id']]
    test_xy['predicted_score'] = predict
    test_xy.sort_values(['predicted_score'],ascending = False,inplace = True)
    # 返回
    return test_xy
    
if __name__ == '__main__':
    # 线下训练集、线下测试集、线上训练集、线上测试集
    print('模型3:')
    print('    预处理...')
    app_launch_log,user_activity_log,user_register_log,video_create_log = load_data()
    print('    构造训练集、测试集...',end = '')
    off_tr,va,on_tr,te = get_dataset(app_launch_log,user_activity_log,user_register_log,video_create_log)
    # 线上训练集->线上测试集
    print('    开始训练...')
    predict = xgb_for_te(on_tr,te)
    print('完毕!')
    # 训练结果
    predict.to_csv(r'tmp/model_3.csv',index = False,header = None)
