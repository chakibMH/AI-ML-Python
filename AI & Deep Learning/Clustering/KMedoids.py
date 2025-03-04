# -*- coding: utf-8 -*-
import pandas as pd
from pyclustering.cluster.kmedoids import kmedoids
import numpy as np
import pickle
from clust_fct import moy_evaluation

def Kmeoid(data, initial_medoids):
    # Create instance of K-Medoids algorithm.
    kmedoids_instance = kmedoids(data, initial_medoids)

    # Run cluster analysis and obtain results.
    kmedoids_instance.process()
    c = kmedoids_instance.get_clusters()

    # Show allocated clusters.
    medoids = kmedoids_instance.get_medoids()
    
    
    for i in (0,1,2):
        for j in range(len(c[i])):
            c[i][j] += 1
            
    #les clusters de chaque utilisateur
    index = c[0] + c[1] + c[2]
    data = [0]*len(c[0]) + [1]*len(c[1]) + [2]*len(c[2])
    
    clusters = pd.DataFrame(index = index, data = {'cluster': data })
    
    return clusters,medoids,c


def start_kmedoid(data_path,pmf_path,formule = 'user-user',nb_clusters = 3):
    
    uim = pd.read_csv(data_path)
    
    uim = uim.pivot_table(values = "rating",index = "userId",
                          columns = "movieId")

    uim.fillna(0,inplace = True)
    
    pmf = pd.read_pickle(pmf_path)
    
    pmf.columns.name = 'movieId'
    pmf.index.name = 'userId'
    
    if formule == 'item-item':
        pivot_pmf = pmf.T
        data = pivot_pmf.values
    elif formule == 'user-user':
        data = pmf.values  
    else:
        raise ValueError("valeur de formule est soit \
                         \'user\-user' ou \'item-item\'")    
    
    
    initial_medoids = (np.random.choice(list(range(len(data))),
                              nb_clusters))
    
    clusters,medoids,c = Kmeoid(data,initial_medoids)
    



    
    moy_p,moy_r = moy_evaluation(uim,clusters,formule,matrice = pmf)


    #enregistrement
    name = "kmedoid_"+formule+"_moy_prec_.pkl"
    with open(name,'wb') as f:
        p = pickle.Pickler(f)
        p.dump(moy_p)
        
    name = "kmedoid_"+formule+"_moy_rec_.pkl"   
    with open(name,'wb') as f:
        p = pickle.Pickler(f)
        p.dump(moy_r)


    
    return moy_p,moy_r

pmf_path = "prediction_matrix_simpleMF_MovieLens100.pkl"

moy_p,moy_r = start_kmedoid("..\\data\\ratings.csv",pmf_path)