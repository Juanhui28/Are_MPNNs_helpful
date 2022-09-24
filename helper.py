import numpy as np, sys, os, random, pdb, json, uuid, time, argparse
from pprint import pprint
import logging, logging.config
from collections import defaultdict as ddict
from ordered_set import OrderedSet
import math
# PyTorch related imports
import torch
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
from torch.utils.data import DataLoader
from torch.nn import Parameter
from torch_scatter import scatter_add

np.set_printoptions(precision=4)

def set_gpu(gpus):
	"""
	Sets the GPU to be used for the run

	Parameters
	----------
	gpus:           List of GPUs to be used for the run
	
	Returns
	-------
		
	"""
	
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def get_logger(name, log_dir, config_dir):
	"""
	Creates a logger object

	Parameters
	----------
	name:           Name of the logger file
	log_dir:        Directory where logger file needs to be stored
	config_dir:     Directory from where log_config.json needs to be read
	
	Returns
	-------
	A logger object which writes to both file and stdout
		
	"""
	config_dict = json.load(open( config_dir + 'log_config.json'))
	config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger

def get_combined_results(left_results, right_results):
	results = {}
	count   = float(left_results['count'])

	results['left_mr']	= round(left_results ['mr'] /count, 5)
	results['left_mrr']	= round(left_results ['mrr']/count, 5)
	results['right_mr']	= round(right_results['mr'] /count, 5)
	results['right_mrr']	= round(right_results['mrr']/count, 5)
	results['mr']		= round((left_results['mr']  + right_results['mr']) /(2*count), 5)
	results['mrr']		= round((left_results['mrr'] + right_results['mrr'])/(2*count), 5)

	for k in range(10):
		results['left_hits@{}'.format(k+1)]	= round(left_results ['hits@{}'.format(k+1)]/count, 5)
		results['right_hits@{}'.format(k+1)]	= round(right_results['hits@{}'.format(k+1)]/count, 5)
		results['hits@{}'.format(k+1)]		= round((left_results['hits@{}'.format(k+1)] + right_results['hits@{}'.format(k+1)])/(2*count), 5)
	return results

def get_param(shape):
	param = Parameter(torch.Tensor(*shape)); 	
	xavier_normal_(param.data)
	return param

def com_mult(a, b):
	r1, i1 = a[..., 0], a[..., 1]
	r2, i2 = b[..., 0], b[..., 1]
	return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim = -1)

def conj(a):
	a[..., 1] = -a[..., 1]
	return a

def cconv(a, b):
	return torch.irfft(com_mult(torch.rfft(a, 1), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

def ccorr(a, b):
	return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

def generate_noise_edge(edge_index, edge_type, num_ent, num_rel, noise_aggre, all_noise, data_dir, data_name):
		
	num_edges = edge_index.size(1) // 2
	in_index, out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
	in_type,  out_type  = edge_type[:num_edges], 	 edge_type [num_edges:]

	noise_num = math.floor(num_edges*noise_aggre)
	print('noise: ', noise_num)

	

	##################### generate noise
	# noise_sr = np.random.randint(num_ent, size=noise_num)
	# noise_ob = np.random.randint(num_ent, size=noise_num)

	# noise_rel_l2r = np.random.randint(num_rel, size=noise_num)
	# noise_rel_inv = noise_rel_l2r + num_rel 
	
	# noise_triplet_in = np.stack((noise_sr, noise_ob), axis = 1)
	# noise_triplet_out = np.stack((noise_ob, noise_sr), axis = 1)
	

	# noise_triplet_in = torch.from_numpy(noise_triplet_in)
	# noise_triplet_out = torch.from_numpy(noise_triplet_out)
	
	# noise_rel_l2r = torch.from_numpy(noise_rel_l2r)
	# noise_rel_inv = torch.from_numpy(noise_rel_inv)
	# # #######################

	# noise_out = open(os.path.join('output', 'random_edge_noise_'+str(noise_aggre)+'.txt'), 'w')

	# for i in range(noise_num):
	# 	noise_out.write(str(noise_triplet_in[i][0].item()) + ' ' + str(noise_rel_l2r[i].item())+ ' ' +str(noise_triplet_in[i][1].item()) + '\n')
		
	# 	noise_out.write(str(noise_triplet_out[i][0].item()) + ' ' + str(noise_rel_inv[i].item())+ ' '+str(noise_triplet_out[i][1].item()) + '\n')
		
	# 	noise_out.flush()
	#####################
	
	noise_triplet_in, noise_triplet_out = [], []
	noise_rel_l2r, noise_rel_inv = [], []
	noise_in = open(os.path.join(data_dir, data_name, 'noise','random_edge_noise_'+str(noise_aggre)+'.txt'), 'r')
	print('path: ', data_name, os.path.join(data_dir, data_name, 'noise','random_edge_noise_'+str(noise_aggre)+'.txt'))
	
	count = 1
	for line in noise_in:
		s, r, o = line.strip().split(' ')
		s, r, o = int(s), int(r), int(o)

		if count % 2 != 0:
			noise_triplet_in.append((s, o))
			noise_rel_l2r.append(r)

		else:

			noise_triplet_out.append((s, o))
			noise_rel_inv.append(r)

		count += 1

	noise_triplet_in = torch.tensor(noise_triplet_in)
	noise_triplet_out = torch.tensor(noise_triplet_out)

	noise_rel_l2r = torch.tensor(noise_rel_l2r)
	noise_rel_inv = torch.tensor(noise_rel_inv)


	if all_noise == 0:
		

		in_index = torch.cat((in_index, noise_triplet_in.t()),dim=1)
		out_index = torch.cat((out_index, noise_triplet_out.t()), dim=1)
			
		in_type = torch.cat((in_type, noise_rel_l2r), dim=0)
		out_type = torch.cat((out_type, noise_rel_inv), dim=0)
		
		edge_index = torch.cat((in_index, out_index), dim=1)
		edge_type = torch.cat((in_type, out_type), dim=0)
		
		print('not all noise')



		return edge_index, edge_type 


	#################### all noise edges

	if all_noise == 1:

		edge_index = torch.cat(( noise_triplet_in.t(), noise_triplet_out.t()), dim=1)
		
		edge_type = torch.cat((noise_rel_l2r, noise_rel_inv), dim=0)
		print('all noise !!')

		return edge_index, edge_type 

def construct_adj(data, num_ent, num_rel, noise_aggre, all_noise, data_dir, data_name):
	"""
	Constructor of the runner class

	Parameters
	----------
	
	Returns
	-------
	Constructs the adjacency matrix for GCN
	
	"""
	edge_index, edge_type = [], []

	for sub, rel, obj in data['train']:
		
		
		edge_index.append((sub, obj))
		
		edge_type.append(rel)
	

	# Adding inverse edges
	
	for sub, rel, obj in data['train']:
		
		edge_index.append((obj, sub))
		edge_type.append(rel + num_rel)

	edge_index	= torch.LongTensor(edge_index).t()
	edge_type	= torch.LongTensor(edge_type)
	
	print('original number of edges:', edge_index.size(1)/2, edge_type.size(0)/2)
	################ adding noisy edges
	if noise_aggre > 0:

		print('################### adding aggregation noises #################')
		edge_index, edge_type = generate_noise_edge(edge_index, edge_type, num_ent, num_rel, noise_aggre, all_noise, data_dir, data_name)
		
		print('noise rate: ', noise_aggre)
		print('number of edges with noises/all noises in aggregation: ', edge_index.size(1), edge_type.size(0))
		print('############### done adding aggregation noises ################')
		print('\n')
	
	return edge_index, edge_type

def get_batch_nhop_neighbors_all( no_partial_2hop, unique_train, node_neighbors, nbd_size=2):
	batch_source_triples = []
	# print("length of unique_entities ", len(unique_train))
	count = 0
	for source in unique_train:
		# randomly select from the list of neighbors
		if source in node_neighbors.keys():
			nhop_list = node_neighbors[source][nbd_size]

			for i, tup in enumerate(nhop_list):
				if(not no_partial_2hop and i >= 2):
					break

				count += 1
				batch_source_triples.append([source, nhop_list[i][0][-1], nhop_list[i][0][0],
												nhop_list[i][1][0]])

	return np.array(batch_source_triples).astype(np.int32)

def read_neighbor_2hop(node_neighbors, ent2id, rel2id, unique_train, para):
	neighor_2hop = {}

	no_partial_2hop = para.no_partial_2hop
	
	# print('2 hop neighbor number before: ', len(node_neighbors))
	for source in node_neighbors.keys():
		nhop_list = node_neighbors[source][2]

		neighor_2hop[ent2id[source]] = {}
		neighor_2hop[ent2id[source]][2] = []
		for i, tup in enumerate(nhop_list):
			relations_1 = tup[0][0]
			relations_2 = tup[0][1]
			
			ent_1 = tup[1][0]
			ent_2 = tup[1][1]
			
			relation = [rel2id[relations_1], rel2id[relations_2]]
			ent = [ent2id[ent_1], ent2id[ent_2]]

			neighor_2hop[ent2id[source]][2].append([relation, ent])

	# print('2 hop neighbor number after: ', len(neighor_2hop))

	indices_2hop = get_batch_nhop_neighbors_all(no_partial_2hop, unique_train, neighor_2hop, nbd_size=2)

	# print('2 hop neighbor num: ', len(indices_2hop))
	indices_2hop = torch.LongTensor(indices_2hop)

	return indices_2hop


def read_feature(embedding_dict, ent2id, rel2id):
	ent_embedding_dict = embedding_dict[0]
	rel_embedding_dict = embedding_dict[1]

	

	ent_num, rel_num = len(ent2id), len(rel2id)//2
	for k in ent_embedding_dict.keys():
		ent_dim = len(ent_embedding_dict[k])
		break
	
	for k in rel_embedding_dict.keys():
		rel_dim = len(rel_embedding_dict[k])
		break

	
	# print('ent: ', len(ent_embedding_dict))
	# print('rel: ', len(rel_embedding_dict))

	feature_embedding = {}
	feature_embedding['entity_embedding'] = torch.zeros((ent_num, ent_dim))
	feature_embedding['relation_embedding'] = torch.zeros((rel_num, rel_dim))

	for k in ent_embedding_dict.keys():
		idx = ent2id[k]
		feature_embedding['entity_embedding'][idx] = torch.tensor(ent_embedding_dict[k])
	
	for k in rel_embedding_dict.keys():
		idx = rel2id[k]
		feature_embedding['relation_embedding'][idx] = torch.tensor(rel_embedding_dict[k])

	
	return feature_embedding
	