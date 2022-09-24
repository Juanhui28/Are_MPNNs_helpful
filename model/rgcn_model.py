from helper import *

from torch import nn
import torch.nn.functional as F
from model.rgcn_conv import RGCNConv
from model.torch_rgcn_conv import FastRGCNConv

class RGCNModel(torch.nn.Module):
	def __init__(self, edge_index, edge_type, params):
		super(RGCNModel, self).__init__()

		self.p = params
		self.edge_index = edge_index
		self.edge_type = edge_type
		self.act	= torch.tanh
		num_rel = self.p.num_rel

		self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim

		self.init_embed		= get_param((self.p.num_ent,   self.p.init_dim))

		# self.init_rel = get_param(( edge_type.unique().size(0), self.p.embed_dim))

		self.init_rel = get_param(( num_rel*2, self.p.init_dim))
		self.w_rel 		= get_param((self.p.init_dim, self.p.embed_dim))

		self.drop = torch.nn.Dropout(self.p.hid_drop)
		

		if self.p.gcn_layer == 1: self.act = None
		self.rgcn_conv1 = RGCNConv(self.p.init_dim, self.p.gcn_dim, self.p.num_rel, self.p.rgcn_num_bases, self.p.rgcn_num_blocks, act=self.act)

		self.rgcn_conv2 = RGCNConv(self.p.gcn_dim, self.p.embed_dim, self.p.num_rel, self.p.rgcn_num_bases, self.p.rgcn_num_blocks) if self.p.gcn_layer == 2 else None

	

	def forward(self, feature=None):
		
		self.edge_index = self.edge_index.to(self.init_rel.device)
		self.edge_type = self.edge_type.to(self.init_rel.device)

		
		x = self.init_embed
		r = self.init_rel

		if feature != None:
			
			x = feature['entity_embedding']
			x = x.to(self.init_rel.device)
			
			if 'relation_embedding' in feature:
				r = Parameter(feature['relation_embedding'])
				r = r.to(self.init_rel.device)
				
		
		x = self.rgcn_conv1(x, self.edge_index, self.edge_type)
		x = self.drop(x)

		
		x = self.rgcn_conv2(x, self.edge_index, self.edge_type) if self.p.gcn_layer == 2 else x
		x = self.drop(x) if self.p.gcn_layer == 2 else x

	
		
		r = torch.matmul(r, self.w_rel)
		# print('rel weight:', self.w_rel)
		# print('rel weight norm', torch.norm(self.w_rel, p=2), (self.w_rel).mean(), self.w_rel.std())
		
		return x, r

		

