#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 11:16:20 2022

@author: antonomaz
"""


import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import sklearn
import json
import glob
import matplotlib.pyplot as plt
from collections import OrderedDict


def lire_fichier (chemin):
    with open(chemin) as json_data: 
        texte =json.load(json_data)
    return texte

def nomfichier(chemin):
    nomfich= chemin.split("/")[-1]
    nomfich= nomfich.split(".")
    nomfich= ("_").join([nomfich[0],nomfich[1]])
    return nomfich
    
import csv


def iob2_to_csv(input_file):
    ligne = []
    with open(input_file, 'r', encoding='utf-8') as f:
        sreader = csv.reader(f,delimiter = " ",quotechar = "\n")
        for line in sreader:
            ligne.append(line)
    return ligne
entites = []
grouped_entities = []
for chemin in glob.glob("DATA/*/*/*.bio"):
    tokens = iob2_to_csv(chemin)
    # print(tokens)
    # break

    for token in tokens:
        if len(token)<2:
            continue
        if token[1]!= 'O' and token[1]!= "":
            entites.append(token)
    #print(entites)
   
    for ent in entites:
        entitie = ent[0]
        grouped_entities.append(entitie)
        #print(grouped_entities)
    
        
    
    file_name = nomfichier(chemin)
    json_file_path = f"{file_name}.json"  
   
    with open(json_file_path, 'w') as f:
        json.dump(grouped_entities, f) 



for path in glob.glob('clustering_08032024/json_pour_cluster/*.json'):
    #path = "clustering_08032024/json_pour_cluster/DAUDET_petit-chose_TesseractFra-PNG.txt_SEM_WiNER.ann_SEM.json-concat.json"
    #path = "clustering_08032024/json_pour_cluster/CARRAUD_petite-Jeanne_TesseractFra-PNG.txt_SEM_WiNER.ann_SEM.json-concat.json"
    #        print("PATH*****",path)
    
    nom_fichier = nomfichier(path)
    #        print(nom_fichier)
    liste=lire_fichier(path)
    
    
    #### FREQUENCE ########
    
    dic_mots={}
    i=0
    
    
    for mot in liste: 
        
        if mot not in dic_mots:
            dic_mots[mot] = 1
        else:
            dic_mots[mot] += 1
    
    i += 1
    
    new_d = OrderedDict(sorted(dic_mots.items(), key=lambda t: t[0]))
    
    freq=len(dic_mots.keys())
    
    
    Set_00 = set(liste)
    Liste_00 = list(Set_00)
    dic_output = {}
    liste_words=[]
    matrice=[]
    cluster_freq = []
    term_counts = []
    
    for l in Liste_00:
            
        if len(l)!=1:
            liste_words.append(l)
    
    
    try:
        words = np.asarray(liste_words)
        for w in words:
            liste_vecteur=[]
        
                
            for w2 in words:
            
                    V = CountVectorizer(ngram_range=(2,3), analyzer='char')
                    X = V.fit_transform([w,w2]).toarray()
                    distance_tab1=sklearn.metrics.pairwise.cosine_distances(X)            
                    liste_vecteur.append(distance_tab1[0][1])
                
            matrice.append(liste_vecteur)
        matrice_def=-1*np.array(matrice)
       
              
        affprop = AffinityPropagation(affinity="precomputed", damping= 0.6, random_state = None) 
    
        affprop.fit(matrice_def)
        
        # 创建一个空列表，用于存储每个聚类的词汇数量
        cluster_word_counts = []
        for cluster_id in np.unique(affprop.labels_):
            exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
            cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
            cluster_str = ", ".join(cluster)
            cluster_list = cluster_str.split(", ")
            
                        
            Id = "ID "+str(i)
            for cle, dic in new_d.items(): 
                if cle == exemplar:
                    dic_output[Id] ={}
                    dic_output[Id]["Centroïde"] = exemplar
                    dic_output[Id]["Freq. centroide"] = dic
                    dic_output[Id]["Termes"] = cluster_list
                    # 记录每个聚类的词汇数量
                    cluster_word_counts.append(len(cluster_list))
            
            i=i+1
            print(dic_output)
            
            
        
    
        with open(f"{nom_fichier}.json",'w') as f:
            json.dump(dic_output,f, indent = 4)
         
        
       
    



    except :        
        print("**********Non OK***********", path)


    #liste_nom_fichier.append(path)
    
    
    # 提取中心词、中心词频率和所包含的词汇数量
    centroids = [entry["Centroïde"] for entry in dic_output.values()]
    centroid_freqs = [entry["Freq. centroide"] for entry in dic_output.values()]
    term_counts = [len(entry["Termes"]) for entry in dic_output.values()]
    
    # 绘制散点图
    plt.scatter(centroids, centroid_freqs, s=term_counts)
    
    # 添加标题和标签
    plt.title(f'{nom_fichier}')
    plt.xlabel('Centroids')
    plt.ylabel('Centroid Frequencies')
    plt.xticks(rotation=90)
    # 显示图形
    plt.show()
    plt.savefig(f"{nom_fichier}.png")