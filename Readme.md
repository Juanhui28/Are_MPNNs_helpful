
<h1 align="center">Are Graph Neural Networks Really Helpful for Knowledge Graph Completion?</h1>
Source code for the ACL paper [Are Graph Neural Networks Really Helpful for Knowledge Graph Completion?](https://arxiv.org/pdf/2205.10652.pdf)

### Dependencies

The key packages are the pytorch,torch-geometric/torch-scatter package. More details of the installation can be refered in https://pytorch.org/ and https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

-  Python 3.7
- ordered-set==4.0.2
- torch=1.4.0+cu100
- torch-cluster=1.5.4
- torch-geometric=1.4.3
- torch-scatter=2.0.4
- torch-sparse=0.6.1
- torch-spline-conv=1.2.0



### Input data
We put the data in this link: https://drive.google.com/file/d/1vQXyMP5HeM-UJBOyRDnXO28YbuZMZeLb/view?usp=sharing

Please decompress the data.zip and put the data folder in the same folder position with the run.py 

- FB15k
- FB15k-237
- WN18
- WN18RR
- NELL-995

### Training model:
commands to run the model:

```shell
########################  CompGCN  ######################

#### compgcn nell-995
python run.py -model 'compgcn' -read_setting 'no_negative_sampling' -neg_num 0  -score_func 'conve' -opn 'corr' -data 'NELL-995'   -lr 0.001 -batch 512  -num_workers 3 -gcn_layer 1 -hid_drop 0.3  

#### compgcn fb15k-237
python run.py -model 'compgcn' -read_setting 'no_negative_sampling' -neg_num 0  -score_func 'conve' -opn 'corr' -data 'FB15k-237'   -lr 0.001 -batch 512  -num_workers 3 -gcn_layer 1 -hid_drop 0.3  

#### compgcn wn18rr
python run.py -model 'compgcn' -read_setting 'no_negative_sampling' -neg_num 0  -score_func 'conve' -opn 'corr' -data 'WN18RR'   -lr 0.001 -batch 512  -num_workers 3 -gcn_layer 1 -hid_drop 0.3  

########################  RGCN  ######################
##rgcn nell-995
python run.py   -model 'rgcn' -read_setting 'negative_sampling' -neg_num 10  -score_func 'distmult' -data 'NELL-995' -rgcn_num_blocks 100  -lr 0.001 -batch 512  -l2 0. -num_workers 3 -gcn_layer 2 -hid_drop 0.1 

##rgcn fb15k-237
python run.py   -model 'rgcn' -read_setting 'negative_sampling' -neg_num 10  -score_func 'distmult' -data 'FB15k-237' -rgcn_num_blocks 100  -lr 0.001 -batch 512  -l2 0. -num_workers 3 -gcn_layer 2 -hid_drop 0.1 

##rgcn wn18rr
python run.py   -model 'rgcn' -read_setting 'negative_sampling' -neg_num 10  -score_func 'distmult' -data 'WN18RR' -rgcn_num_blocks 100  -lr 0.001 -batch 512  -l2 0. -num_workers 3 -gcn_layer 2 -hid_drop 0.1 

##rgcn wn18
python run.py   -model 'rgcn' -read_setting 'negative_sampling' -neg_num 10  -score_func 'distmult' -data 'WN18' -rgcn_num_blocks 100  -lr 0.00001 -batch 512  -l2 0. -num_workers 3 -gcn_layer 2 -hid_drop 0.05 

##rgcn fb15k
python run.py   -model 'rgcn' -read_setting 'negative_sampling' -neg_num 10  -score_func 'distmult' -data 'FB15k' -rgcn_num_blocks 100  -lr 0.0001 -batch 512  -l2 0. -num_workers 3 -gcn_layer 1 -hid_drop 0.1  -init_dim 600 -gcn_dim 600 -k_w 10 -k_h 60 -no_edge_reverse


########################  KBGAT  ######################
##kbgat fb15k-237
python run.py -model 'kbgat'  -read_setting 'no_negative_sampling' -neg_num 0 -score_func 'conve' -data 'FB15k-237'  -lr 0.001 -batch 512  -l2 0. -num_workers 3  -hid_drop 0.3 

##kbgat wn18rr 
python run.py  -model 'kbgat'  -read_setting 'no_negative_sampling' -neg_num 0 -score_func 'conve' -data 'WN18RR'  -lr 0.001 -nheads 8 -batch 512  -l2 0. -num_workers 3  -hid_drop 0.3

## kbgat nell 
python  run.py  -model 'kbgat'  -read_setting 'no_negative_sampling' -neg_num 0  -score_func 'conve' -data 'NELL-995'  -lr 0.001 -nheads 8 -batch 512  -l2 0. -num_workers 3  -hid_drop 0.3  

########################  mlp  ######################
## compgcn-mlp / kbgat-mlp: nell-995
python run.py  -model 'mlp' -read_setting 'no_negative_sampling' -neg_num 0 -score_func 'conve' -data 'NELL-995'  -lr 0.001 -batch 512  -l2 0. -num_workers 3 -gcn_layer 1 -hid_drop 0.3 

## compgcn-mlp / kbgat-mlp: fb15k237
python run.py  -model 'mlp' -read_setting 'no_negative_sampling' -neg_num 0 -score_func 'conve' -data 'FB15k-237'  -lr 0.001 -batch 512  -l2 0. -num_workers 3 -gcn_layer 1 -hid_drop 0.3  

## compgcn-mlp / kbgat-mlp: wn18rr
python run.py  -model 'mlp' -read_setting 'no_negative_sampling' -neg_num 0 -score_func 'conve' -data 'WN18RR'  -lr 0.001 -batch 512  -l2 0. -num_workers 3 -gcn_layer 1 -hid_drop 0.3  



####rgcn-mlp fb15k-237
python run.py -model 'mlp' -read_setting 'negative_sampling'  -neg_num 10  -score_func 'distmult' -data 'FB15k-237' -lr 0.0001 -batch 512  -l2 0. -num_workers 3 -gcn_layer 2 -hid_drop 0.1 

####rgcn-mlp wn18
python run.py -model 'mlp' -read_setting 'negative_sampling'  -neg_num 10  -score_func 'distmult' -data 'WN18' -lr 0.0001 -batch 512  -l2 0. -num_workers 3 -gcn_layer 2 -hid_drop 0.05 
  
####rgcn-mlp fb15k
python run.py -model 'mlp' -read_setting 'negative_sampling'  -neg_num 10  -score_func 'distmult' -data 'FB15k' -lr 0.0001 -batch 512  -l2 0. -num_workers 3 -gcn_layer 1 -hid_drop 0.1 -init_dim 600 -gcn_dim 600 -k_w 10 -k_h 60 
```

### Citation:
Please cite the following paper if you use this code in your work.
```bibtex
@article{li2022graph,
  title={Are Graph Neural Networks Really Helpful for Knowledge Graph Completion?},
  author={Li, Juanhui and Shomer, Harry and Ding, Jiayuan and Wang, Yiqi and Ma, Yao and Shah, Neil and Tang, Jiliang and Yin, Dawei},
  journal={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, {ACL} 2023, Toronto, Cananda},
  year={2023}
}
```
For any clarification, comments, or suggestions please create an issue or contact [Juanhui](https://github.com/Juanhui28).
