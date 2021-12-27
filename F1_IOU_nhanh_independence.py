#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 18:21:20 2021

@author: vinh
"""
import os
import numpy as np
from natsort import natsorted
from imutils import paths
from PIL import Image
import pdb
import time
import collections
import cv2 
import wandb

start = time.time()
#----------------------------------------------------------------
def intersection_union(y_true,y_pred):
    intersection=(y_pred*y_true).sum()
    total=(y_pred+y_true).sum()
    union=total-intersection
    return intersection, union

def tp_fp_fn(y_true, y_pred):
    tp = (y_true * y_pred).sum()
    tn = ((1 - y_true) * (1 - y_pred)).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()   
    return tp, fp, fn

def metric_save(prediction, gt):
    #--------------------caculate metric------------------------------------
    gt[gt>=100]=1 # convert to binary image
    gt[gt!=1]=0 # convert to binary image
    #gt=np.squeeze(gt)
    
    #pdb.set_trace()

    intersection, union= intersection_union(gt, prediction)
    tp, fp, fn=tp_fp_fn(gt, prediction)
 
    #----------------------------------------------------------------------       
    return intersection, union, tp, fp, fn
#-------------------------------------------

def cal_metrics(final_predict="./result_bw",gt="./mask_full_size/"):
    #"./laynguong/OTSU_GAUS"#    
    dataPaths_raw=natsorted(list(paths.list_images(final_predict)))
    dataPaths_gt=natsorted(list(paths.list_images(gt)))
    test_lst_raw=[dataPath.strip()for dataPath in dataPaths_raw]
    test_lst_gt=[dataPath.strip()for dataPath in dataPaths_gt]
    
    
    
    intersection_save=[]
    #total_save=[]
    union_save=[]  
    save_tp=[]
    save_fp=[]
    save_fn=[]
    
    save_f1_all=[]
    save_iou_all=[]
    
    save_name=[]
    
    for idx in range(0, len(test_lst_raw)):#len(test_lst_raw)
        
        #pdb.set_trace()
        y_pred=cv2.imread(test_lst_raw[idx],0)
        #y_pred = np.array(y_pred)
        #y_pred=list(y_pred.flatten())
        if np.max(y_pred)>127:
            y_pred[y_pred<=127.5]=0
            y_pred[y_pred>127.5]=1
        else:
            y_pred[y_pred<=0.5]=0
            y_pred[y_pred>0.5]=1
        
        name_gt=test_lst_gt[idx]#.replace(test_lst_gt[idx].split('/')[-1], test_lst_raw[idx].split('/')[-1])
        #pdb.set_trace()
        
        
    
        #y_true=Image.open(test_lst_gt[idx]).convert('1')
        #pdb.set_trace()
        #name_gt=name_gt.replace('.png','_maskfg.png')
        y_true=cv2.imread(name_gt,0)
        #pdb.set_trace()
        if np.max(y_true)>127:
            y_true=y_true/255
        
        #y_true = np.array(y_true)
        #y_true=list(y_true.flatten())
        #pdb.set_trace()
        #collections.Counter(y_pred.flatten())
        
        try:
            intersection, union, tp, fp, fn = metric_save(y_pred, y_true)
        except Exception:
            pdb.set_trace()
        
        #pdb.set_trace()
        each_f1,  each_iou = 2*tp/(2*tp+fp+fn), intersection/union 
        print(test_lst_raw[idx].split("/")[-1]+f"--idx:{idx}---f1:{each_f1}---iou:{each_iou}")
        # if(each_iou<0.1):
        #     pdb.set_trace()
        #wandb.log({'each_f1':each_f1}, step=idx) 
        #wandb.log({'each_iou':each_iou}, step=idx)  
        save_name.append(f'--idx:{idx}--name:{test_lst_raw[idx]}--each_f1:{each_f1}--each_iou:{each_iou}')

        save_f1_all.append(each_f1) 
        save_iou_all.append(each_iou)           
            
        
                   
        intersection_save.append(intersection)
        union_save.append(union)        
        save_tp.append(tp)
        save_fp.append(fp)
        save_fn.append(fn)
        
    mean_f1, mean_iou = 2*sum(save_tp)/(2*sum(save_tp)+sum(save_fp)+sum(save_fn)), sum(intersection_save)/sum(union_save)            
    end = time.time()
    interval=end - start
    print('---time is:',interval)   
    print('-- Average fscore:',mean_f1)
    print('-- Average iou:',mean_iou)
    
    mean_all_f1=sum(save_f1_all)/len(save_f1_all)
    mean_all_iou=sum(save_iou_all)/len(save_iou_all)
    print('-- mean_all_f1:',mean_all_f1)
    print('-- mean_all_iou:',mean_all_iou)    

    method_name=final_predict.split('/')[-2]
    with open(f'log_{method_name}.txt', 'w') as f:
        for idx in range(0, len(save_name)):
            f.writelines("%s\n" % save_name[idx])
            if idx==(len(save_name)-1):
                f.writelines("%s\n" % f'-- mean_all_f1:{mean_all_f1}--mean_all_iou:{mean_all_iou}')
    
    return mean_all_f1, mean_all_iou
    
if __name__ == '__main__':
    
    #img_f_fld='./backup/1keersc1/105/testset_full/'
    #img_f_fld='./backup/1keersc1/105/testset_fusion/'  
    
    #img_f_fld ='./preds/Text/RGB_VST_testset_full'
    
    img_f_fld ='./result_bw/'
    
    #img_f_fld ='./result_textnet/'
    
    graph_name = img_f_fld.split('/')[-2]    
    
    wandb.init(project='tinh_metric', name=graph_name)   
    mean_f1,mean_iou= cal_metrics(final_predict=img_f_fld, gt="./mask_full_size/")    
    wandb.log({'mean_f1':mean_f1}, step=1038) 
    wandb.log({'mean_iou':mean_iou}, step=1038)    
    print(img_f_fld)
   
