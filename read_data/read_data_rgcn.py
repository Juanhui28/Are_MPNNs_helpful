from .data_loader import TrainDataset, TestDataset
import gzip
import random
import math
import torch
from ordered_set import OrderedSet
from collections import defaultdict as ddict
from torch.utils.data import DataLoader
from helper import *
import pickle

class read_rgcn(object):


	def __init__(self, params, triplet_reverse_loss=False):
		self.p = params
		self.reverse = triplet_reverse_loss
		# edge_index, edge_type = self.load_data()

	

	def load_data(self):
		

		ent_set, rel_set = OrderedSet(), OrderedSet()
	
		for split in ['train', 'test', 'valid']:
			print((self.p.data_dir+'/{}/{}.txt'.format(self.p.dataset, split)))
			for line in open(self.p.data_dir+'/{}/{}.txt'.format(self.p.dataset, split)):
				# print(line.strip().split('\t'))
				if len(line.strip().split('\t')) < 3:
					
					continue
				sub, rel, obj = map(str.lower, line.strip().split('\t'))
				

				ent_set.add(sub)
				rel_set.add(rel)
				ent_set.add(obj)
		
		

		self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
		self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}

		self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})

		self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
		self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}



		self.p.num_ent		= len(self.ent2id)
		self.p.num_rel		= len(self.rel2id) // 2
		self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

		print('dataset: ', self.p.dataset)
		print('number of entities: ', self.p.num_ent)
		print('number of relations: ', self.p.num_rel)
		

		self.data = ddict(list)
		sr2o = ddict(set)
		train_edge_num  = 0
		#######
		
		
		unique_train = set()
		for split in ['train', 'test', 'valid']:
			for line in open(self.p.data_dir+'/{}/{}.txt'.format(self.p.dataset, split)):
				if len(line.strip().split('\t')) < 3:
					continue
				sub, rel, obj = map(str.lower, line.strip().split('\t'))
					
				sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]

				
				self.data[split].append((sub, rel, obj))

				if split == 'train': 

					train_edge_num += 1
					unique_train.add(sub)
					unique_train.add(obj)

					sr2o[(sub, rel)].add(obj) 
					if not self.reverse:
						sr2o[(obj, rel)].add(sub)	
					else:
						sr2o[(obj, rel+self.p.num_rel)].add(sub)

		

		self.data = dict(self.data)

		self.sr2o = {k: list(v) for k, v in sr2o.items()}
		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				sr2o[(sub, rel)].add(obj)

				if not self.reverse:
					sr2o[(obj, rel)].add(sub)
				else:
					sr2o[(obj, rel+self.p.num_rel)].add(sub)


		self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
		self.triples  = ddict(list)

		rel_number = self.p.num_rel
		

		self.edge_index, self.edge_type = construct_adj(self.data, self.p.num_ent, rel_number, self.p.noise_rate, self.p.all_noise, self.p.data_dir, self.p.dataset)
		
		#################### rgcn 
		if self.p.loss_noise_rate > 0:
			num_rel = self.p.num_rel if not self.reverse else self.p.num_rel*2

			self.data['train'] = self.generate_loss_noise(self.p.loss_noise_rate, self.data['train'], train_edge_num, self.p.num_ent, num_rel)

		for (sub, rel, obj) in self.data['train']:
			
			self.triples['train'].append({'triple':(sub, rel, obj),  'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
			
			if not self.reverse:
				self.triples['train'].append({'triple':(obj, rel, sub),  'label': self.sr2o[(obj, rel)], 'sub_samp': 1})
			
			else:
				self.triples['train'].append({'triple':(obj, rel+self.p.num_rel, sub),  'label': self.sr2o[(obj, rel+ self.p.num_rel)], 'sub_samp': 1})
			
		
		for split in ['test', 'valid']:
			
			for sub, rel, obj in self.data[split]:
				
				self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
				
				if not self.reverse:
					self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel, sub), 'label': self.sr2o_all[(obj, rel)]})
				
				else:
					self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel+self.p.num_rel, sub), 'label': self.sr2o_all[(obj, rel+self.p.num_rel)]})
				
		##################### rgcn


		self.triples = dict(self.triples)
		

		def get_data_loader(dataset_class, split, batch_size, shuffle=True):
			return  DataLoader(
					dataset_class(self.triples[split], self.p.num_ent, self.p.neg_num, self.p.use_all_neg_samples, self.p.lbl_smooth),
					batch_size      = batch_size,
					shuffle         = shuffle,
					num_workers     = max(0, self.p.num_workers),
					collate_fn      = dataset_class.collate_fn
				)

		if self.p.use_all_neg_samples == False and self.p.neg_num == 0: 
			
			raise ValueError('negative sampling methods uses the negative sampling when computing loss. Please set the predefined parameter ''neg_num'' larger than 0')
		
		
		

		self.data_iter = {
			'train':    	get_data_loader(TrainDataset, 'train', 	    self.p.batch_size),
			'valid_head':   get_data_loader(TestDataset,  'valid_head', self.p.batch_size),
			'valid_tail':   get_data_loader(TestDataset,  'valid_tail', self.p.batch_size),
			'test_head':   	get_data_loader(TestDataset,  'test_head',  self.p.batch_size),
			'test_tail':   	get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
		}

		
		file = self.p.data_dir + '/'+self.p.dataset+'/2hop_neighbor_myindex.pickle'
		if os.path.exists(file):
			with open(file, 'rb') as handle:
				node_neighbors = pickle.load(handle)
				
			indices_2hop = read_neighbor_2hop(node_neighbors,  self.ent2id, self.rel2id, unique_train, self.p)
		else:
			indices_2hop = None


		if self.p.use_feat_input:
			print('use transe feature!!')
			file = self.p.data_dir + '/'+self.p.dataset+'/feature_embedding.pickle'

			with open(file, 'rb') as handle:
				embedding_dict = pickle.load(handle)

			file = self.p.data_dir + '/'+self.p.dataset+'/2hop_neighbor_myindex.pickle'
			with open(file, 'rb') as handle:
				node_neighbors = pickle.load(handle)
			
			feature_embeddings =read_feature(embedding_dict, self.ent2id, self.rel2id)
			
		else:
			feature_embeddings = None
		
		if not self.p.no_edge_reverse:
			
			print('max rel index: ', max(self.edge_type))
			
			return self.edge_index, self.edge_type, self.data_iter, feature_embeddings, indices_2hop
			
		else:
			
			e = self.edge_index.size(-1) // 2

			print('edge no inverse:', self.edge_index[:, :e].size(), max(self.edge_type[:e]))
			edge_index = self.edge_index[:, :e]
			edge_type = self.edge_type[:e]

			return edge_index, edge_type, self.data_iter, feature_embeddings, indices_2hop
			

	
	def read_batch(self, batch, split, device):
		
		if split == 'train':
			triple, label, pos_neg_ent  = [ _.to(device) for _ in batch]
			
			return triple[:, 0], triple[:, 1], triple[:, 2], label, pos_neg_ent
			
		else:
			triple, label = [ _.to(device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label


	