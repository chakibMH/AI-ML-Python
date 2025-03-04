# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
from item import preprocessItem
from user import preprocessU



def evaluation(votes,pred,k = 10,seuil = 3.5):
 while(votes.shape[0] < k):
    k = int(input("on a moins de {} votes \n veuiilez entrer une nouvelle val de k".format(k)))
 #nb of all relevent items 
 relevent = len(votes.loc[votes >= seuil].index)
 non_relevent = len(votes) - relevent
 #pred = pred.loc[pred >= seuil]
 pred.sort_values(ascending = False,inplace = True)
 #on choisit les k meilleurs
 pred_top_k = pred.iloc[0:k-1].index
 # nb of relevent items in the selected nb_rec items
 relevent_selected = sum([elt in votes.loc[votes >= seuil].index for elt in pred_top_k])
 print('true relevent = ',relevent)
 print('relevent selected  = ',relevent_selected)
 pred.sort_values(inplace = True)
 pred_worst_k = pred.iloc[0:k-1].index
 non_relevent_selected = sum([elt in votes.loc[votes < seuil].index for elt in pred_worst_k])
 print('true NON relevent = ',non_relevent)
 print('relevent selected  = ',non_relevent_selected)
 #precision 1: relevent ;;;; 0:non-relevent
 p1 = relevent_selected/k
 p0 = non_relevent_selected/k
 #recall
 r1 = relevent_selected/relevent if relevent != 0 else 0
 r0 = non_relevent_selected/non_relevent if non_relevent != 0 else 0
 print("-"*10)
 print("k = ",k) 
 print("precision : \n")
 print("  1 : ",p1)
 print("  0 : ",p0)
 print("recall : \n")
 print("  1 : ",r1)
 print("  0 : ",r0)
 print("-"*10)
 return p1,r1,p0,r0
 

def moy_evaluation(uim,clusters,formule,matrice = None,liste_k = [10,15,20],seuil = 3.5, N = 30):
    """retourne la moyenne de rappel et precision de tous les 
    utilisateurs pour tout k dans liste_k"""
    users = uim.index
    moy_precision_rel = defaultdict(float)
    moy_recall_rel = defaultdict(float)
    moy_precision_n_rel = defaultdict(float)
    moy_recall_n_rel = defaultdict(float)
    
    if formule == 'user-user':
        preprocess = preprocessU
    elif formule == 'item-item':
       preprocess = preprocessItem 
    else:
        raise ValueError("valeur de formule est soit \
                         \'user\-user' ou \'item-item\'")
                         
    for u in users:
        votes,pred = preprocess(uim,u,clusters,mat = matrice)
        #calcul du vote pour chaque k dans la liste
        for k in liste_k:
            p1,r1,p0,r0 = evaluation(votes,pred,k,seuil)
            #on fait la somme pour chaque k
            moy_precision_rel[k] += p1
            moy_recall_rel[k] += r1
            moy_precision_n_rel[k] += p0
            moy_recall_n_rel[k] += r0
    #on devise sur taille pour chaque k
    taille = len(users)
    print("\n\n######  moyenne evaluation  #####\n")
    for k in liste_k:
        moy_precision_rel[k] /= taille
        moy_recall_rel[k] /= taille
        moy_precision_n_rel[k] /= taille
        moy_recall_n_rel[k] /= taille
        #affichage
        # print("\n-----------------")
        # print("k = ",k)
        # print("precision moyenne : ",moy_precision_rel)
        # print("rappel moyen : ",moy_recall)
        
        
    moy_p = {1:moy_precision_rel,0:moy_precision_n_rel}
    moy_r = {1:moy_recall_rel,0:moy_recall_n_rel}       
    
    return moy_p,moy_r



def max_moy(uim, liste_k = [10,15,20], seuil= 3.5):
    users = uim.index
    max_moy_precision = defaultdict(float)
    max_moy_recall = defaultdict(float)
    for u in users:
        votes = uim.loc[u].map(lambda x:x if x != 0 else np.nan).copy()
        votes.dropna(inplace = True)
        true_relevent = len(votes.loc[votes >= seuil].index)
        for k in liste_k:

            p_max = min (1, true_relevent / k )
            v = k / true_relevent if true_relevent != 0 else 0
            
            r_max = min (1, v )
            
            max_moy_precision[k] += p_max
            max_moy_recall[k] += r_max
    taille = len(users)
    for k in liste_k:
        max_moy_precision[k] /= taille
        max_moy_recall[k] /= taille
    
    return max_moy_precision,max_moy_recall
# precision max = min( 1, true_relevent / k ) 
# recall max = min( 1, k / true_relevent )