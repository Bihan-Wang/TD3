#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 11:16:20 2022

@author: antonomaz
"""
#%%
#run
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import DistanceMetric 
from sklearn.feature_extraction.text import CountVectorizer
import sklearn
import json
import glob
import re
import os
from collections import OrderedDict


def lire_fichier (chemin):#factoriser pour lire les fichiers json
    with open(chemin) as json_data: 
        texte =json.load(json_data)#ouvre le fichier JSON spécifié, lit son contenu et le stocke dans la variable texte
        # retourne le contenu du fichier JSON sous forme de texte.
    return texte

def nomfichier(chemin):#extrait le nom du fichier
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
    

# def group_entities(entities):
#     grouped_entities = []#pour stocker les resultats
#     current_entity = []#les entites sont en train de executer

#     for token, label in entities:
#         if label.startswith('B-'):
#             if current_entity:
#                 grouped_entities.append(current_entity)
#                 current_entity = []#vide 
#             current_entity.append(token)
#         elif label.startswith('I-'):
#             if current_entity:
#                 current_entity.append(token)
#         # else:
#         #     if current_entity:
#         #         grouped_entities.append(current_entity)
#         #         current_entity = []
#     if current_entity:
#         grouped_entities.append(current_entity)

#     return grouped_entities


# for chemin in glob.glob("DATA/*/*/*.bio"):
#     tokens = iob2_to_csv(chemin)
#     entites = []

#     for token in tokens:
#         if len(token) < 2:
#             continue
#         if token[1] != 'O' and token[1] != "":
#             entites.append(token)

#     # 调用函数进行分组
#     grouped_entities = group_entities(entites)

# # # 打印结果
# for group in grouped_entities:
#     print(group)

# #定义函数将结果保存为 JSON 文件
# def save_entities_to_json(entities, output_file):
#     # 分组实体
#     grouped_entities = group_entities(entities)
#     # 写入 JSON 文件
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(grouped_entities, f, ensure_ascii=False)
#     print("结果已成功写入到", output_file)

# # 遍历文件夹中的每个路径
# for chemin in glob.glob("DATA/*/*/*.bio"):
#     # 读取实体并保存为 JSON
#     tokens = iob2_to_csv(chemin)
#     entites = [token for token in tokens if len(token) >= 2 and token[1] != 'O' and token[1] != ""]
#     save_entities_to_json(entites, os.path.splitext(chemin)[0] + '.json')
#%%
#run




subcorpus = "DATA/NOAILLES/NOAILLES_TesseractFra-PNG/NOAILLES_la-nouvelle-esperance_TesseractFra-PNG.txt.txt"
print("SUBCORPUS***",subcorpus)
liste_nom_fichier =[]#creer une liste pour stocker les noms des fichiers
path = "NOAILLES_la-nouvelle-esperance_Tesseract-PNG_txt_bio.json"
print("PATH*****",path)
        
nom_fichier = nomfichier(path)#extrait les noms des fichiers dans le chemin
#        print(nom_fichier)
liste=lire_fichier(path)#contenir le contenu du fichier JSON
#print(liste)

#### FREQUENCE ########

dic_mots={}#initialiser une dictionnaire pour stocker les resultats de frequence
i=0#pour compter


for mot in liste: #pour compter les frequences des mots dans le contenu du fichier JSON
    
    mot_tuple = tuple(mot)

# 检查元组是否在字典中
    if mot_tuple not in dic_mots:
# 如果不在，则将元组添加到字典并将计数器设置为1
        dic_mots[mot_tuple] = 1
    else:
# 如果已经在字典中，则将计数器加1
        dic_mots[mot_tuple] += 1

i += 1 #pour compter le nombre des mots excutes

new_d = OrderedDict(sorted(dic_mots.items(), key=lambda t: t[0]))#trier les mots par ordre alphabetique

freq=len(dic_mots.keys())#compter le nombre de mots uniques


Set_00 = set(liste)#supprimer les mots repetes et transforme une liste a un set
Liste_00 = list(Set_00)#transforme le set a une liste
dic_output = {}
liste_words=[]
matrice=[]
#%%
#run
for l in liste:
        
    if len(l)!=1:
        liste_words.append(l)


try:
    words = np.asarray(liste_words)#Convertit la liste liste_words en un tableau NumPy words.
    for w in words:
        liste_vecteur=[] #pour tous les mots , creer une liste
    
            
        for w2 in words:
        
                V = CountVectorizer(ngram_range=(2,3), analyzer='char')#convertir les mots en vecteurs
                X = V.fit_transform([w,w2]).toarray()#transformer w et w2 en vecteurs 
                distance_tab1=sklearn.metrics.pairwise.cosine_distances(X)            
                liste_vecteur.append(distance_tab1[0][1])#claculer la distance entre les deux vecteurs
            
        matrice.append(liste_vecteur)#ajouter la liste liste_vecteur à la liste matrice, créant une matrice de similarité.
    matrice_def=-1*np.array(matrice)
   
          
    affprop = AffinityPropagation(affinity="precomputed", damping= 0.6, random_state = None) #Crée un objet de propagation d'affinité affprop, utilisant la matrice de similarité précalculée comme mesure de similarité.

    affprop.fit(matrice_def)#regrouper les données en clusters.
    for cluster_id in np.unique(affprop.labels_):
        exemplar = words[affprop.cluster_centers_indices_[cluster_id]]#identifier les centroides
        cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])# Identifie tous les membres du cluster
        cluster_str = ", ".join(cluster)#Convertit les mots du cluster en une chaîne de caractères séparée par des virgules.
        cluster_list = cluster_str.split(", ")#diviser les chaines et stoker dans une liste
                    
        Id = "ID "+str(i)#un identifiant unique pour chaque cluster.
        for cle, dic in new_d.items(): 
            if cle == exemplar: #Vérifie si l'élément actuel du dictionnaire est l'exemplaire du cluster.
                dic_output[Id] ={}
                dic_output[Id]["Centroïde"] = exemplar#le controide du claster
                dic_output[Id]["Freq. centroide"] = dic# la frequence de controide
                dic_output[Id]["Termes"] = cluster_list #les membres de controide
        
        i=i+1
        print(dic_output)
        break
    

except :        #si il y a quelques erreurs
    print("**********Non OK***********", path)#afficher les faux chemins


    liste_nom_fichier.append(path)
    
    
# continue 


    
