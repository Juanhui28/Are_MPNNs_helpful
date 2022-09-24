from typing import Optional, Union, Tuple


import torch, math
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter as Param
from torch.nn import Parameter
from torch_scatter import scatter
from torch_sparse import SparseTensor, matmul, masked_select_nnz
from model.message_passing import MessagePassing

from helper import *


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def masked_edge_index(edge_index, edge_mask):
	if isinstance(edge_index, Tensor):
		return edge_index[:, edge_mask]
	else:
		return masked_select_nnz(edge_index, edge_mask, layout='coo')


class RGCNConv(MessagePassing):

	def __init__(self, in_channels: int,
					out_channels: int,
					num_relations: int,
					num_bases: Optional[int] = None,
					num_blocks: Optional[int] = None,
					aggr: str = 'mean',
					root_weight: bool = True,
					bias: bool = True, 
					low_mem=False, 
					act=None,
					**kwargs):  # yapf: disable

		super(RGCNConv, self).__init__(aggr=aggr,  **kwargs)

		if num_bases is not None and num_blocks is not None:
			raise ValueError('Can not apply both basis-decomposition and '
								'block-diagonal-decomposition at the same time.')

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.num_relations = num_relations*2
		self.num_bases = num_bases
		self.num_blocks = num_blocks
		self.low_mem = low_mem
		self.act = act
		# 

		if isinstance(in_channels, int):
			in_channels = (in_channels, in_channels)
		self.in_channels_l = in_channels[0]

		if num_bases is not None:
			self.weight = get_param((num_bases, in_channels[0], out_channels))
			self.comp = get_param((self.num_relations, num_bases))
			

		elif num_blocks is not None:
			# print(in_channels[0], num_blocks, out_channels)
			assert (in_channels[0] % num_blocks == 0
					and out_channels % num_blocks == 0)
			self.weight = get_param((self.num_relations, num_blocks,
								in_channels[0] // num_blocks,
								out_channels // num_blocks))
			self.register_parameter('comp', None)

		else:
			self.weight = get_param((self.num_relations, in_channels[0], out_channels))
			self.register_parameter('comp', None)

		if root_weight:
			self.root = get_param((in_channels[1], out_channels))
		else:
			self.register_parameter('root', None)

		if bias:
			
			self.register_parameter('bias', Parameter(torch.zeros(out_channels)))
		else:
			self.register_parameter('bias', None)

		self.bn = torch.nn.BatchNorm1d(out_channels)

		# self.reset_parameters()
		self.invest = 1

	# def reset_parameters(self):
	# 	glorot(self.weight)
	# 	glorot(self.comp)
	# 	glorot(self.root)
	# 	zeros(self.bias)
		


	def forward(self, x, edge_index, edge_type):
		if self.invest == 1:
			print('edge number in the modle: ', edge_index.size(1), edge_type.size(0))
			

		if self.num_bases is not None:
			if self.invest == 1:
				print('use low mem: ', self.low_mem)
				print('regualizer: basis')
			out = self.basic_decom_func(x, edge_index, edge_type)
		
		elif self.num_blocks is not None:
			if self.invest == 1:
				print('use low mem: ', self.low_mem)
				print('regualizer: block')
			out = self.block_decomp_func(x, edge_index, edge_type)
		
		else:
			raise ValueError('only implemnt RGCN with basic decompsition and block diagonal-decomposition')
		
		# print('weight:', self.weight)
		# print('weight norm', torch.norm(self.weight, p=2), (self.weight).mean(), self.weight.std())

		
		self.invest = 0
		return out
	def basic_decom_func(self, x, edge_index, edge_type):
		

		if self.low_mem:
			weight = self.weight
		
			weight = (self.comp @ weight.view(self.num_bases, -1)).view(
			self.num_relations, self.in_channels_l, self.out_channels)
			out = torch.zeros(x.size(0), self.out_channels, device=x.device)

			for i in range(self.num_relations):
				tmp = masked_edge_index(edge_index, edge_type == i)

				h = self.propagate('mean', tmp, x=x, edge_type=edge_type, norm=None)
				out = out + (h @ weight[i])
		else:
			norm = self.compute_norm(edge_type, edge_index, dim_size=x.size(0))

			out = self.propagate('add', edge_index, x=x, edge_type=edge_type, norm=norm)

		root = self.root
		if root is not None:
			out +=  x @ root

		if self.bias is not None:
			out += self.bias
		
		if self.bn is not None:
			out = self.bn(out)
		
		if self.act is not None:
			out = self.act(out)

		
		return out


	
	def block_decomp_func(self, x, edge_index, edge_type):
		

		if self.low_mem:
			weight = self.weight
			out = torch.zeros(x.size(0), self.out_channels, device=x.device)

			for i in range(self.num_relations):
				tmp = masked_edge_index(edge_index, edge_type == i)
				h = self.propagate('mean', tmp, x=x, edge_type=edge_type, norm=None)
				h = h.view(-1, weight.size(1), weight.size(2))
				h = torch.einsum('abc,bcd->abd', h, weight[i])
				out += h.contiguous().view(-1, self.out_channels)
		
		else:
			norm = self.compute_norm(edge_type, edge_index, dim_size=x.size(0))

			out = self.propagate('add', edge_index, x=x, edge_type=edge_type, norm=norm)
			
		
		root = self.root
		if root is not None:
			out +=  x @ root

		if self.bias is not None:
			out += self.bias
		
		if self.bn is not None:
			out = self.bn(out)
		
		if self.act is not None:
			out = self.act(out)

		return out
		

	def message(self, x_j, edge_type, edge_index, norm):
		if self.low_mem:
			return x_j
		
		else:

			if  self.num_bases is not None:
				weight = self.weight
		
				weight = (self.comp @ weight.view(self.num_bases, -1)).view(
					self.num_relations, self.in_channels_l, self.out_channels)

				out = torch.bmm(x_j.unsqueeze(-2), weight[edge_type]).squeeze(-2)

				return out if norm is None else out * norm.view(-1, 1)
				
				

			elif self.num_blocks is not None:
				weight = self.weight
				# weight = weight.index_select(0, edge_type).view(-1, weight.size(2), weight.size(3))
				weight = weight[edge_type].view(-1, weight.size(2), weight.size(3))
				x_j = x_j.view(-1, 1, weight.size(1))
				out = torch.bmm(x_j, weight).view(-1, self.out_channels)

				return out if norm is None else out * norm.view(-1, 1)


	def compute_norm(self,  edge_type: Tensor, index: Tensor,
					dim_size: Optional[int] = None) -> Tensor:

		# Compute normalization in separation for each `edge_type`.
		
		norm = F.one_hot(edge_type, self.num_relations).to(torch.float)
		norm = scatter(norm, index[0], dim=0, dim_size=dim_size)[index[0]]
		norm = torch.gather(norm, 1, edge_type.view(-1, 1))
		norm = 1. / norm.clamp_(1.)	

		return norm
		# return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size)	
