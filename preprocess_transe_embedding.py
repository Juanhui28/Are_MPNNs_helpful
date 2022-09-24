# %%
import os
import numpy as np
from sklearn import datasets
import torch
import pickle


def init_embeddings(entity_file, relation_file):
    entity_emb, relation_emb = [], []

    with open(entity_file) as f:
        for line in f:
            entity_emb.append([float(val) for val in line.strip().split()])

    with open(relation_file) as f:
        for line in f:
            relation_emb.append([float(val) for val in line.strip().split()])

    # return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)
    return entity_emb, relation_emb



def read_entity_from_id(filename='./data/WN18RR/entity2id.txt'):
    entity2id = {}
    id2entity = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                entity, entity_id = line.strip().split()[0].strip(), line.strip().split()[1].strip()
                entity2id[entity] = int(entity_id)
                id2entity[int(entity_id)] = entity

    return id2entity


def read_relation_from_id(filename='./data/WN18RR/relation2id.txt'):
    relation2id = {}
    id2relation = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                relation, relation_id = line.strip().split()[0].strip(), line.strip().split()[1].strip()

                relation2id[relation] = int(relation_id)
                id2relation[int(relation_id)] = relation

    return id2relation

# %%
#################### generate feature embedding
dataset = 'NELL-995'

id2ent = read_entity_from_id('data_kbgat/'+dataset+'/entity2id.txt')
id2rel = read_relation_from_id('data_kbgat/'+dataset+'/relation2id.txt')

entity_emb, relation_emb = init_embeddings('data_kbgat/'+dataset+'/entity2vec.txt', 'data_kbgat/'+dataset+'/relation2vec.txt')

print('entity number: ', len(id2ent), len(entity_emb))
print('relation number: ', len(id2rel), len(relation_emb))

ent_embedding_dict= {}
rel_embedding_dict= {}

# %%

for i in range(len(id2ent)):
    ent = id2ent[i]
    # print('ent dim: ', len( entity_emb[i]))
    ent_embedding_dict[ent] = entity_emb[i]

for i in range(len(id2rel)):
    rel = id2rel[i]
    # print('rel dim: ', len( relation_emb[i]))
    rel_embedding_dict[rel] = relation_emb[i]

print('entity number: ', len(ent_embedding_dict))
print('relation number: ', len(rel_embedding_dict))


feature_embedding = [ent_embedding_dict, rel_embedding_dict]

file = 'data_kbgat/' + dataset+ "/feature_embedding.pickle"
with open(file, 'wb') as handle:
    pickle.dump(feature_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)


################# 2hop

# %%
#################### generate 2 hop neighbors

file =  'data_kbgat/'+dataset+'/2hop.pickle'

id2ent = read_entity_from_id('data_kbgat/'+dataset+'/entity2id.txt')
id2rel = read_relation_from_id('data_kbgat/'+dataset+'/relation2id.txt')

with open(file, 'rb') as handle:
    node_neighbors = pickle.load(handle)

print('len: ', len(node_neighbors))
neighor_2hop = {}

for source in node_neighbors.keys():
    nhop_list = node_neighbors[source][2]

    neighor_2hop[id2ent[source]] = {}
    neighor_2hop[id2ent[source]][2] = []

    for i, tup in enumerate(nhop_list):
        relations_1 = tup[0][0]
        relations_2 = tup[0][1]
        
        ent_1 = tup[1][0]
        ent_2 = tup[1][1]
        
        relation = [id2rel[relations_1], id2rel[relations_2]]
        ent = [id2ent[ent_1], id2ent[ent_2]]

        neighor_2hop[id2ent[source]][2].append([relation, ent])

print('len new: ', len(neighor_2hop))
file = 'data_kbgat/' + dataset+ "/2hop_neighbor_myindex.pickle"
with open(file, 'wb') as handle:
    pickle.dump(neighor_2hop, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%
