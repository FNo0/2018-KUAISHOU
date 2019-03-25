# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 04:32:21 2018

@author: zjy
"""

import pandas as pd
import numpy as np
import pickle
import re
from functools import reduce 
import datetime
import xgboost as xgb

import model_1
import model_2
import model_3
import model_4

import warnings
warnings.filterwarnings('ignore')

def run_model_1():
    # 线下训练集、线下测试集、线上训练集、线上测试集
    print('模型1:')
    print('    预处理...')
    app_launch_log,user_activity_log,user_register_log,video_create_log = model_1.load_data()
    print('    构造训练集、测试集...',end = '')
    off_tr,va,on_tr,te = model_1.get_dataset(app_launch_log,user_activity_log,user_register_log,video_create_log)
    # 线上训练集->线上测试集
    print('    开始训练...')
    predict = model_1.xgb_for_te(on_tr,te)
    print('完毕!')
    print()
    # 训练结果
    predict.to_csv(r'tmp/model_1.csv',index = False,header = None)
    # 返回
    return predict

def run_model_2():
    # 线下训练集、线下测试集、线上训练集、线上测试集
    print('模型2:')
    print('    预处理...')
    source = model_2.load_data()
    print('    构造训练集、测试集...',end = '')
    off_tr,va,on_tr,te = model_2.get_dataset(source)
    # 线上训练集->线上测试集
    print('    开始训练...')
    predict = model_2.xgb_for_te(on_tr,te)
    print('完毕!')
    print()
    # 训练结果
    predict.to_csv(r'tmp/model_2.csv',index = False,header = None)
    # 返回
    return predict

def run_model_3():
    # 线下训练集、线下测试集、线上训练集、线上测试集
    print('模型3:')
    print('    预处理...')
    app_launch_log,user_activity_log,user_register_log,video_create_log = model_3.load_data()
    print('    构造训练集、测试集...',end = '')
    off_tr,va,on_tr,te = model_3.get_dataset(app_launch_log,user_activity_log,user_register_log,video_create_log)
    # 线上训练集->线上测试集
    print('    开始训练...')
    predict = model_3.xgb_for_te(on_tr,te)
    print('完毕!')
    print()
    # 训练结果
    predict.to_csv(r'tmp/model_3.csv',index = False,header = None)
    # 返回
    return predict

def run_model_4():
    # 线上训练集、线上测试集
    print('模型4:')
    print('    预处理...')
    app_launch, user_activity, user_register, video_create = model_4.load_data()
    print('    构造训练集、测试集...', end='')
    train,test= model_4.get_dataset(app_launch, user_activity, user_register, video_create)
    # 线上训练集->线上测试集
    print('    开始训练...')
    predict = model_4.xgb_for_te(train,test)
    print('完毕!')
    print()
    # 训练结果
    predict.to_csv(r'tmp/model_4.csv', index=False, header=None)
    # 返回
    return predict

def ronghe_jiaoji_prob(model1,model2,w1,w2):
    '''
    交集+概率融合
    1.model1和model2的预测概率按照w1和w2的权重相加
    2.求model1和model2排名前25000个user的交集
    3.交集个数肯定不足25000，按照概率大小补全不重复的user
    
    '''
    # model_1
    m1 = model1.copy()
    m1.rename(columns = {'predicted_score' : 'm1'},inplace = True)
    maxs = max(m1['m1'].tolist())
    mins = min(m1['m1'].tolist())
    m1['m1'] = m1['m1'].map(lambda x : (x - mins) / (maxs - mins))
    # model_2
    m2 = model2.copy()
    m2.rename(columns = {'predicted_score' : 'm2'},inplace = True)
    maxs = max(m2['m2'].tolist())
    mins = min(m2['m2'].tolist())
    m2['m2'] = m2['m2'].map(lambda x : (x - mins) / (maxs - mins))
    # 交集融合
    result1 = pd.merge(m1[['user_id']].head(25000),m2[['user_id']].head(25000),on = ['user_id'],how = 'inner')
    # 概率融合
    result2 = pd.merge(m1,m2,on = ['user_id'],how = 'left')
    result2['predicted_score'] = result2['m1'] * w1 + result2['m2'] * w2
    result2.sort_values(['predicted_score'],ascending = False,inplace = True)
    result2 = result2[['user_id']]
    # 交集概率融合
    result = pd.concat([result1,result2],axis = 0)
    result.drop_duplicates(['user_id'],keep = 'first',inplace = True)
    # 取前25000个
    result = result.head(25000)
    # 返回
    return result

if __name__ == '__main__':
    ## 程序开始运行
    starttime = datetime.datetime.now()
    # 四个模型的预测概率
    result1 = run_model_1()
    result2 = run_model_2()
    result3 = run_model_3()
    result4 = run_model_4()
    # result1分别和result2,result3,result4按照交集+概率融合
    result12 = ronghe_jiaoji_prob(result1,result2,0.8,0.2)
    result13 = ronghe_jiaoji_prob(result1,result3,0.7,0.3)
    result14 = ronghe_jiaoji_prob(result1,result4,0.6,0.4)
    # 三个融合结果取交集
    result = pd.merge(result12,result14,how = 'inner',on = 'user_id')
    result = pd.merge(result,result13,how = 'inner',on = 'user_id')
    result.to_csv(r'result/result.csv',index = False,header = None) # 最终结果！！！
    ## 程序结束运行
    endtime = datetime.datetime.now()
    print()
    print('运行时间:',end = '')
    print((endtime - starttime).seconds / 60,end = '')
    print('分钟。')
    print()
    print('整体运行完毕,请前往result文件夹查看。')
