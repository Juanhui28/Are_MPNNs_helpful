# from model.compgcn_model import CompGCN_TransE, CompGCN_DistMult, CompGCN_ConvE
from model.compgcn_model import CompGCNBase
from model.rgcn_model import RGCNModel
from model.KBGAT import KBGAT_Model
from helper import *
import torch
import torch.nn.functional as F

class mlp(torch.nn.Module):

	def __init__(self, in_channels, out_channels, act, params):
		super(mlp, self).__init__()

		self.p = params
		self.act	= act

		self.W_entity	= get_param((in_channels, out_channels))
		self.W_relation = get_param((in_channels, out_channels))
		self.bn	= torch.nn.BatchNorm1d(out_channels)

		self.drop = torch.nn.Dropout(self.p.hid_drop)
		if self.p.bias:  self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

	def forward(self, x, r):
		out =  torch.mm(x, self.W_entity)

		if self.p.bias: out = out + self.bias

		out = self.bn(out)
		if self.act is not None:
			out = self.act(out)


		return out, torch.matmul(r, self.W_relation)	


class BaseModel(torch.nn.Module):
	def __init__(self, edge_index, edge_type, params, feature_embeddings=None, indices_2hop=None):
		super(BaseModel, self).__init__()

		self.p		= params
		
		#### loss
		self.bceloss	= torch.nn.BCELoss()   ##bce loss
		self.logsoftmax_loss = torch.nn.CrossEntropyLoss(reduction='mean')
		self.margin = self.p.margin   ####for margin loss



		self.edge_index = edge_index
		self.edge_type = edge_type
		self.act = torch.tanh
	
		
		self.inter_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim

		if self.p.model == 'random' and self.p.score_func.lower() == 'conve':
			if self.p.init_dim != self.p.embed_dim:
				self.p.init_dim = self.p.embed_dim

		self.init_embed		= get_param((self.p.num_ent,   self.p.init_dim))
		
		if self.p.score_func == 'transe': 	self.init_rel = get_param((self.p.num_rel,   self.p.init_dim))
		else: 					self.init_rel = get_param((self.p.num_rel*2, self.p.init_dim))

		# self.init_rel = get_param((edge_type.unique().size(0), self.p.init_dim))

		if self.p.model == 'compgcn':
			
			self.model = CompGCNBase(self.edge_index, self.edge_type, self.p)
			

		elif self.p.model == 'rgcn':
			
			self.model = RGCNModel(self.edge_index, self.edge_type, self.p)
			# self.init_rel = get_param(( self.p.num_rel, self.p.init_dim))
		
		elif self.p.model == 'kbgat':
			
			self.model = KBGAT_Model(self.edge_index, self.edge_type, self.p, feature_embeddings, indices_2hop)


		self.TransE_score = TransE_score(self.p)
		self.DistMult_score = DistMult_score(self.p)
		self.ConvE_score = ConvE_score(self.p)

		self.mlp1 = mlp(self.p.init_dim, self.inter_dim, self.act, self.p)
		self.mlp2 = mlp(self.inter_dim, self.p.embed_dim, self.act, self.p) if self.p.gcn_layer == 2 else None

		self.drop = torch.nn.Dropout(self.p.hid_drop)
		self.invest = 1


	def loss(self, pred, true_label, original_score = None, pos_neg_ent=None):


		if self.p.loss_func == 'bce':

			if self.invest == 1:
				print('loss function: BCE')

			pred = torch.sigmoid(pred)
			loss = self.bceloss(pred, true_label)
			
		
		
			
		self.invest = 0
		return loss, pred

	def get_loss(self, x, r, sub, rel, label, pos_neg_ent):

		sub_emb	= torch.index_select(x, 0, sub)
		rel_emb	= torch.index_select(r, 0, rel)

		if self.p.score_func.lower() == 'transe':
			if self.invest == 1:
				print('score function: transe')
				
			score, original_score = self.TransE_score(sub_emb, rel_emb, x, label, pos_neg_ent)
		
		elif self.p.score_func.lower() == 'distmult':
			if self.invest == 1:
				print('score function: distmult')

			score , original_score= self.DistMult_score(sub_emb, rel_emb, x, label, pos_neg_ent)
		
		elif self.p.score_func.lower() == 'conve':
			if self.invest == 1:
				print('score function: conve')
				
			# print('sss', sub_emb.size(), rel_emb.size(), x.size(), r.size(), sub.size(), rel.size())
			score, original_score = self.ConvE_score(sub_emb, rel_emb, x, label, pos_neg_ent)


		
		loss, score = self.loss(score, label, original_score, pos_neg_ent)
		

		return score, loss


	def forward(self, feature=None):

		if self.p.model == 'random':

			if self.invest == 1:
				
				print('investigation mode: random initialization')
				if feature != None:
					print('use feature as input!!!!')
			
			r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
			x = self.init_embed

			x	= self.drop(x)

		
		elif  self.p.model == 'mlp':
			if self.invest == 1:
				print('investigation mode: mlp')
				if feature != None:
					print('use feature as input!!!!')
		
			r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
			x = self.init_embed

			x, r = self.mlp1(x, r)
			x	= self.drop(x)

			x, r = self.mlp2(x, r) if self.p.gcn_layer == 2 else (x, r)
			x	= self.drop(x) if self.p.gcn_layer == 2 else x
		


		else:
			if self.invest == 1:
				print('investigation mode: aggregation')
				
			x, r = self.model()


		return x, r


class TransE_score(torch.nn.Module):
	def __init__(self, params=None):
		super(TransE_score, self).__init__()
		self.p = params


	def forward(self, sub, rel, x, label,  pos_neg_ent=None):

		sub_emb, rel_emb, all_ent	= sub, rel, x
		
		obj_emb				= sub_emb + rel_emb


		x	= self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)		
		original_score = x

		if pos_neg_ent != None:
			
			row_idx = torch.arange(x.size()[0], device=x.device)
			row_idx = row_idx.unsqueeze(1).repeat(1,pos_neg_ent.size(-1))
			x = x[row_idx, pos_neg_ent]

		
	
		return x, original_score

class DistMult_score(torch.nn.Module):
	def __init__(self, params=None):
		super(DistMult_score, self).__init__()
		self.p = params
		
		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
		
	def forward(self, sub, rel, x, label, pos_neg_ent=None):

		sub_emb, rel_emb, all_ent	= sub, rel, x
		obj_emb				= sub_emb * rel_emb
		
		
		x = torch.mm(obj_emb, all_ent.transpose(1, 0))
		x += self.bias.expand_as(x)
		original_score = x
		

		if pos_neg_ent != None:
			
			row_idx = torch.arange(x.size()[0], device=x.device)
			row_idx = row_idx.unsqueeze(1).repeat(1,pos_neg_ent.size(-1))
			x = x[row_idx, pos_neg_ent]

		
		return x, original_score

class ConvE_score(torch.nn.Module):
	def __init__(self,  params=None):
		super(ConvE_score, self).__init__()
		self.p = params
		self.bn0		= torch.nn.BatchNorm2d(1)
		self.bn1		= torch.nn.BatchNorm2d(self.p.num_filt)
		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)
		
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
		self.m_conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

		flat_sz_h		= int(2*self.p.k_w) - self.p.ker_sz + 1
		flat_sz_w		= self.p.k_h 	    - self.p.ker_sz + 1
		self.flat_sz		= flat_sz_h*flat_sz_w*self.p.num_filt
		self.fc			= torch.nn.Linear(self.flat_sz, self.p.embed_dim)

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

	def concat(self, e1_embed, rel_embed):
		e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
		rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
		return stack_inp

	def forward(self, sub, rel, x, label, pos_neg_ent=None):

		sub_emb, rel_emb, all_ent	= sub, rel, x
		stk_inp				= self.concat(sub_emb, rel_emb)
		x				= self.bn0(stk_inp)
		x				= self.m_conv1(x)
		x				= self.bn1(x)
		x				= F.relu(x)
		x				= self.feature_drop(x)
		x				= x.view(-1, self.flat_sz)
		x				= self.fc(x)
		x				= self.hidden_drop2(x)
		x				= self.bn2(x)
		x				= F.relu(x)

		x = torch.mm(x, all_ent.transpose(1,0))
		x += self.bias.expand_as(x)
		original_score = x
		

		if pos_neg_ent != None:
			
			row_idx = torch.arange(x.size()[0], device=x.device)
			row_idx = row_idx.unsqueeze(1).repeat(1,pos_neg_ent.size(-1))
			x = x[row_idx, pos_neg_ent]

		
		
		return x, original_score
