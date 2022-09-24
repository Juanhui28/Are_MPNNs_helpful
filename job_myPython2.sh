#!/bin/bash
# 系统环境变量 wiki: http://wiki.baidu.com/pages/viewpage.action?pageId=1053742013#2.4%20CaaS
current_rank_index=${POD_INDEX}
rank_0_ip=${POD_0_IP}
free_port=${TRAINER_PORTS}
dist_url="tcp://${rank_0_ip}:${free_port}"
world_size=2
echo "current_rank_index: ${current_rank_index}"
echo "dist_url: ${dist_url}"
echo "world_size: ${dist_url}"

lsof -i:${free_port}

function switch_py36() {
    # switch python3
    export PY36_HOME=/opt/conda/envs/py36

    export PATH=${PY36_HOME}/bin:$PATH
    export LD_LIBRARY_PATH=${PY36_HOME}/lib/:${LD_LIBRARY_PATH}
}

function py27_local() {
    # default py27
    echo "this is py27 local train job..."
    python main.py -a resnet18 ./afs/imagenet2012
}
function install_packages(){
    pip install tqdm
    pip install tensorboardX
    pip install ./thirdparty/torch_cluster-latest+cu101-cp36-cp36m-linux_x86_64.whl
    pip install ./thirdparty/torch_scatter-latest+cu101-cp36-cp36m-linux_x86_64.whl
    pip install ./thirdparty/torch_sparse-latest+cu101-cp36-cp36m-linux_x86_64.whl
    pip install ./thirdparty/torch_spline_conv-latest+cu101-cp36-cp36m-linux_x86_64.whl
    pip install pytest-runner
    pip install pytest-cov
    pip install ./thirdparty/torch_geometric-1.4.3.tar.gz
}
function use_my_python(){
    export PYTHON_HOME=${TRAIN_WORKSPACE}/env_run/thirdparty/pytorch1.7_cuda11.0
    export LD_LIBRARY_PATH=${PYTHON_HOME}/lib:${LD_LIBRARY_PATH}
    export PATH=${PYTHON_HOME}/bin:${PATH}

    pip install ordered_set
}
function use_my_python_cuda10(){
    export PYTHON_HOME=${TRAIN_WORKSPACE}/env_run/thirdparty/pytorch1.6_cuda10.1
    export LD_LIBRARY_PATH=${PYTHON_HOME}/lib:${LD_LIBRARY_PATH}
    export PATH=${PYTHON_HOME}/bin:${PATH}

    pip install ordered_set
}

function py36_local() {
    echo " this is py36 local train job..."
    #####################
    # switch_py36
    # # pytorch 1.3.0 需要降低 Pillow 的版本， pytorch 1.4.0 无此问题
    # pip install Pillow==6.2.2
    # install_packages
    #####################
    use_my_python
    # use_my_python_cuda10

    

    ##############################rgcn 237
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237' -invest_mode 'aggre' -kill_cnt 20 -noise_rate 0 -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.3 -init_dim 100   -k_w 10 -k_h 20 
   
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237' -invest_mode 'aggre' -kill_cnt 20 -seed 2 -noise_rate 0 -neg_num 10  -use_all_neg_samples  -rgcn_num_blocks 100  -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.3 -init_dim 100   -k_w 10 -k_h 20  
    
    
  
    #########################rgcn fb15k
    #-no_edge_reverse
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k' -invest_mode 'aggre' -kill_cnt 20  -noise_rate 0 -neg_num 10 -rgcn_num_blocks 100   -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 600  -gcn_dim 600 -k_w 10 -k_h 60  
    
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k' -invest_mode 'mlp'  -kill_cnt 20  -noise_rate 0 -neg_num 10 -rgcn_num_blocks 100 -lr 0.01 -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 600  -gcn_dim 600  -k_w 10 -k_h 60  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k' -invest_mode 'score'    -noise_rate 0 -neg_num 10 -rgcn_num_blocks 100 -lr 0.0001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.2  -init_dim 600   -k_w 10 -k_h 20  
    
    ####################rgcn wn18
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18' -invest_mode 'aggre' -kill_cnt 20 -noise_rate 0 -seed 2 -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.00001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.05 -init_dim 100 -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18' -invest_mode 'mlp'  -kill_cnt 20 -noise_rate 0 -neg_num 10 -rgcn_num_blocks 100 -lr 0.0001 -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18' -invest_mode 'score'  -seed 2 -kill_cnt 20 -noise_rate 0 -neg_num 10 -rgcn_num_blocks 100 -lr 0.001 -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.05  -init_dim 100   -k_w 10 -k_h 20  
    
    ####################rgcn wn18rr

    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18RR' -invest_mode 'aggre'  -kill_cnt 20  -noise_rate 0 -all_noise 0 -neg_num 10 -rgcn_num_blocks 100 -lr 0.001  -no_edge_reverse -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  
    


    ####################rgcn  + add noise in aggre
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18RR' -invest_mode 'aggre'   -kill_cnt 20  -noise_rate 5 -all_noise 0 -neg_num 10 -rgcn_num_blocks 100 -lr 0.0001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237' -invest_mode 'aggre'   -kill_cnt 20   -noise_rate 5 -all_noise 0. -neg_num 10 -rgcn_num_blocks 100 -lr 0.001  -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  
    

    ###################single relation
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18/single_relation' -invest_mode 'aggre' -kill_cnt 20 -noise_rate 0 -all_noise 0 -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.00001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.05  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237/single_relation' -invest_mode 'aggre' -kill_cnt 20 -noise_rate 0 -all_noise 0  -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.01  -batch 512  -l2 1e-4 -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  
    
#    python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18/single_relation' -invest_mode 'mlp' -seed 2 -kill_cnt 20 -noise_rate 0 -all_noise 0 -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.00001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.05  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237/single_relation' -invest_mode 'mlp' -kill_cnt 20  -noise_rate 0 -all_noise 0 -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.5  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18RR/single_relation' -invest_mode 'aggre' -kill_cnt 20 -seed 2 -noise_rate 0 -all_noise 0 -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.00001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18RR/single_relation' -invest_mode 'mlp' -kill_cnt 20   -noise_rate 0 -all_noise 0 -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.0001  -batch 512  -l2 1e-7 -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  
    
   
   ##############################noise in aggregation: single relation
#    python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18/single_relation' -invest_mode 'aggre' -kill_cnt 20 -noise_rate 2 -all_noise 0. -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.05  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237/single_relation' -invest_mode 'aggre'  -kill_cnt 20 -seed 2 -noise_rate 5 -all_noise 0. -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.01  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18RR/single_relation' -invest_mode 'aggre' -kill_cnt 20  -seed 2 -noise_rate 5 -all_noise 0. -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.00001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  
    

    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18/single_relation' -invest_mode 'mlp' -seed 2 -kill_cnt 20 -noise_rate 1 -all_noise 1 -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.00001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.05  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237/single_relation' -invest_mode 'mlp' -kill_cnt 20 -noise_rate 1 -all_noise 0. -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  
    
    
    
   
    ###################remove single relation
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18/remove_single_relation' -invest_mode 'aggre' -kill_cnt 20 -seed 2 -noise_rate 0 -all_noise 0 -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.0001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.05  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237/remove_single_relation' -invest_mode 'aggre'  -kill_cnt 20 -noise_rate 0 -all_noise 0 -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237/remove_single_relation' -invest_mode 'mlp'  -seed 2 -kill_cnt 20 -noise_rate 0 -all_noise 0 -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18RR/remove_single_relation' -invest_mode 'aggre' -kill_cnt 20  -seed 2  -noise_rate 0 -all_noise 0 -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.0001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18RR/remove_single_relation' -invest_mode 'mlp' -kill_cnt 20  -seed 2 -noise_rate 0 -all_noise 0 -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  
    
    ###################remove single relation + noise in aggregation
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237/remove_single_relation' -invest_mode 'aggre'  -kill_cnt 20 -seed 2 -noise_rate 1 -all_noise  0 -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18RR/remove_single_relation' -invest_mode 'aggre' -kill_cnt 20  -seed 1 -noise_rate 1 -all_noise 1 -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  
    

    ######################## remove single relation + max density
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18RR/remove_single_relation_maxdensity' -invest_mode 'aggre' -kill_cnt 20  -seed 2 -noise_rate 0 -all_noise 0 -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.0001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18RR/remove_single_relation_maxdensity' -invest_mode 'mlp' -kill_cnt 20 -seed 2 -noise_rate 0 -all_noise 0 -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.0001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  
    
    ######################## remove single relation + max density + noise in aggre
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18RR/remove_single_relation_maxdensity' -invest_mode 'aggre' -kill_cnt 20 -seed 2 -noise_rate 1 -all_noise 1. -neg_num 10 -no_edge_reverse  -rgcn_num_blocks 100 -lr 0.0001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  
    

    ######################## gnn dataset
    ##with feat
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'pubmed' -invest_mode 'aggre' -use_feat_input -feat_type 'gnn_feature' -one_rel  -read_gnn_data -neg_num 10  -rgcn_num_blocks 100 -lr 0.0001   -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 500   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'pubmed' -invest_mode 'mlp' -use_feat_input -feat_type 'gnn_feature' -one_rel -read_gnn_data -neg_num 10  -rgcn_num_blocks 100 -lr 0.001   -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.1 -init_dim 500   -k_w 10 -k_h 20  
    

    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'citeseer' -invest_mode 'aggre' -use_feat_input -feat_type 'gnn_feature' -one_rel  -read_gnn_data -neg_num 10  -rgcn_num_blocks 100 -lr 0.0001   -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 3703   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'citeseer' -invest_mode 'mlp' -use_feat_input -feat_type 'gnn_feature' -one_rel  -read_gnn_data -neg_num 10  -rgcn_num_blocks 100 -lr 0.01   -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 3703   -k_w 10 -k_h 20  
    
    ### noise + feat
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'pubmed' -invest_mode 'aggre' -use_feat_input -feat_type 'gnn_feature' -one_rel -read_gnn_data -neg_num 10 -noise_rate 1 -all_noise 1. -rgcn_num_blocks 100 -lr 0.0001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 500   -k_w 10 -k_h 20  
    
    ###no feat
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'pubmed' -invest_mode 'aggre' -one_rel -neg_num 10 -read_gnn_data -rgcn_num_blocks 100 -lr 0.0001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 500   -k_w 10 -k_h 20  
    
    python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'pubmed' -invest_mode 'mlp'  -one_rel -neg_num 10 -read_gnn_data -rgcn_num_blocks 100 -lr 0.0001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 500   -k_w 10 -k_h 20  
    
    ### noise + nofeat
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'pubmed' -invest_mode 'aggre' -one_rel -neg_num 10 -read_gnn_data -noise_rate 1 -all_noise 1. -rgcn_num_blocks 100 -lr 0.0001   -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 500   -k_w 10 -k_h 20  
    
}    


function py36_local_multicard() {
    # pytorch spawn 只支持 py3.4 或者个更高，因此该实例单机多卡，仅跑 py3
    echo "this is py36 local multicard train job..."
    ########################
    # switch_py36
    # install_packages
    # # pytorch 1.3.0 需要降低 Pillow 的版本， pytorch 1.4.0 无此问题
    # pip install Pillow==6.2.2
    #########################
    use_my_python
    # use_my_python_cuda10
    # python run.py  -data_dir './afs' -output_dir './output' -score_func 'transe' -opn 'sub' 

    #python main.py -a resnet50 --dist-url ${dist_url} --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./afs/imagenet2012

}

function py36_distribute() {
    # pytorch spawn 只支持 py3.4 或者个更高，因此该实例单机多卡，仅跑 py3
    echo "this is py36 distribute train job..."
    switch_py36
    # pytorch 1.3.0 需要降低 Pillow 的版本， pytorch 1.4.0 无此问题
    pip install Pillow==6.2.2
    python main.py -a resnet50 \
                    --dist-url ${dist_url} \
                    --dist-backend 'nccl' \
                    --multiprocessing-distributed \
                    --world-size ${world_size} \
                    --rank ${current_rank_index} ./afs/imagenet2012
}

function main() {
    if [[ "${IS_STANDALONE}" = "1" ]]; then
        echo "this is local mode job, will run py27 and py36 local train job..."
        if [[ "${TRAINER_GPU_CARD_COUNT}" = "1" ]]; then
            echo "this one gpu card train job..."
            # py27_local
            # sleep 2
            py36_local
        else
            echo "this multi gpu card train job..."
            py36_local_multicard
        fi
    else
        echo " this is distribute train job..."
        py36_distribute
    fi
    echo "finished!"
}

main
