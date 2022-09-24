from helper import *
from torch.utils.data import Dataset
import random
import torch



class TrainDataset(Dataset):
	
	def __init__(self, triples, num_ent, neg_num, use_all_neg_samples, lbl_smooth):
		self.triples	= triples
		self.num_ent = num_ent
		self.neg_num = neg_num
		self.use_all_neg_samples = use_all_neg_samples
		self.lbl_smooth = lbl_smooth
		self.entities	= np.arange(self.num_ent, dtype=np.int32)

		if self.use_all_neg_samples:
			self.neg_num = self.num_ent - 1
		
		
		

	def __len__(self):
		return len(self.triples)


	def __getitem__(self, idx):
		ele			= self.triples[idx]
		triple, label, sub_samp	= torch.LongTensor(ele['triple']), np.int32(ele['label']), np.float32(ele['sub_samp'])
		
		trp_label		= self.get_label(label)
		num = self.num_ent

		if self.neg_num > 0:
			pos_neg_ent, trp_label = self.get_neg_ent(triple, trp_label)
			trp_label = torch.FloatTensor(trp_label)
			num = self.neg_num + 1
		else:
			pos_neg_ent = torch.tensor([0])
		
	
		if self.lbl_smooth != 0.0:
			trp_label = (1.0 - self.lbl_smooth)*trp_label + (1.0/num)


		return triple, trp_label, pos_neg_ent

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		trp_label	= torch.stack([_[1] 	for _ in data], dim=0)
		pos_neg_ent = torch.stack([_[2] 	for _ in data], dim=0)
	

		return triple, trp_label, pos_neg_ent
	
	def get_neg_ent(self, triple, label):
		def get(triple, label):
			pos_obj		= triple.reshape(1,-1)
			mask		= np.ones([self.num_ent], dtype=np.bool)
			label       = np.array(label.numpy(), dtype=bool)
			
			mask[label]	= 0
			
			if self.neg_num <= self.entities[mask].shape[0]:
				neg_ent		= np.int32(np.random.choice(self.entities[mask], self.neg_num, replace=False)).reshape([-1])
				

			else:
				more_neg_num = self.neg_num - self.entities[mask].shape[0]
				neg_ent		= np.int32(np.random.choice(self.entities[mask], more_neg_num, replace=False)).reshape([-1])

				neg_ent =  np.concatenate((self.entities[mask], neg_ent) )
				
			


			neg_triplet = np.tile(pos_obj, (self.neg_num, 1))
			neg_triplet[:, 2] = neg_ent
			pos_neg_triplet = np.concatenate((pos_obj, neg_triplet))

			
			
			pos_neg_ent = np.concatenate((triple[2].numpy().reshape(-1), neg_ent))
			# print('pos: ',triple[2] )
			# print('pos_neg: ', pos_neg_ent)
			

			all_label = np.zeros( self.neg_num + 1, dtype=np.float32)
			all_label[0] = 1

			return torch.LongTensor(pos_neg_ent), all_label

		pos_neg_ent, all_label = get(triple, label)
		return pos_neg_ent, all_label

	def get_label(self, label):
		y = np.zeros([self.num_ent], dtype=np.float32)
		for e2 in label: y[e2] = 1.0

		return torch.FloatTensor(y)


class TestDataset(Dataset):
	
	def __init__(self, triples,  num_ent, neg_num=None, use_all_neg_samples=False, lbl_smooth=None):
		self.triples	= triples
		self.num_ent =  num_ent
		

	def __len__(self):
		return len(self.triples)

	
	def __getitem__(self, idx):
		ele		= self.triples[idx]
		triple, label	= torch.LongTensor(ele['triple']), np.int32(ele['label'])
		label		= self.get_label(label)
		

		return triple, label

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		label		= torch.stack([_[1] 	for _ in data], dim=0)

		
		return triple, label
	
	def get_label(self, label):
		y = np.zeros([self.num_ent], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)