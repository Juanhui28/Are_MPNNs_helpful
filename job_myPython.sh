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

    
	#############################original compgcn

    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult' -opn 'mult' -data 'FB15k-237' -invest_mode 'aggre' -neg_num 0 -noise_rate 0 -lr 0.001 -batch 128  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult' -opn 'mult' -data 'FB15k-237' -invest_mode 'mlp' -neg_num 0 -noise_rate 0 -lr 0.001 -batch 128  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult' -opn 'mult' -data 'FB15k-237' -invest_mode 'score' -neg_num 0 -noise_rate 0 -lr 0.001 -batch 128  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  
   


    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'conve' -opn 'corr' -data 'FB15k-237' -invest_mode 'aggre'   -neg_num 0  -noise_rate 0 -seed 2 -lr 0.0001 -batch 128  -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  
   
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'conve' -opn 'corr' -data 'FB15k-237' -invest_mode 'score'  -neg_num 0  -noise_rate 0 -lr 0.001 -batch 128  -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.3  -init_dim 200   -k_w 10 -k_h 20  
    


    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'transe' -opn 'sub' -data 'FB15k-237' -invest_mode 'aggre' -neg_num 0 -noise_rate 0 -lr 0.001 -batch 128  -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  -gamma 9  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'transe' -opn 'sub' -data 'FB15k-237' -invest_mode 'mlp' -neg_num 0 -noise_rate 0 -lr 0.0001 -batch 128  -l2 0. -num_workers 10 -gcn_layer 1 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  -gamma 40  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'transe' -opn 'sub' -data 'FB15k-237' -invest_mode 'score' -neg_num 0 -noise_rate 0 -lr 0.0001 -batch 128  -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.1  -init_dim 200   -k_w 10 -k_h 20  -gamma 30  
   

    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult' -opn 'sub' -data 'WN18' -invest_mode 'aggre' -neg_num 0 -noise_rate 0 -lr 0.0001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult' -opn 'sub' -data 'WN18RR' -invest_mode 'aggre' -seed 1 -neg_num 0 -noise_rate 0 -lr 0.0001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.2  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult' -opn 'sub' -data 'WN18RR' -invest_mode 'mlp' -neg_num 0 -noise_rate 0 -seed 2 -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  
    

    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'conve' -opn 'corr' -data 'WN18RR' -invest_mode 'aggre' -seed 2 -neg_num 0 -noise_rate 0 -lr 0.01 -batch 512  -l2 0. -num_workers 10 -gcn_layer 1 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'conve' -opn 'corr' -data 'WN18RR' -invest_mode 'mlp' -seed 2 -neg_num 0 -noise_rate 0 -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'conve' -opn 'corr' -data 'WN18RR' -invest_mode 'score' -neg_num 0 -noise_rate 0 -lr 0.0001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  
    
    
    ##############################rgcn 237
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237' -invest_mode 'aggre' -kill_cnt 20 -noise_rate 0 -neg_num 10 -rgcn_num_blocks 100  -seed 2 -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.3 -init_dim 100   -k_w 10 -k_h 20 
    
    ### use all nodes as negative samples
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237' -invest_mode 'aggre' -kill_cnt 20 -seed 2 -noise_rate 0 -neg_num 1  -use_all_neg_samples  -rgcn_num_blocks 100  -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.3 -init_dim 100   -k_w 10 -k_h 20  
    
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237' -invest_mode 'mlp'  -kill_cnt 20 -noise_rate 0 -neg_num 5 -seed 2 -lr 0.0001 -batch 512   -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.2  -init_dim 100   -k_w 10 -k_h 20  
     ### use all nodes as negative samples
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237' -invest_mode 'mlp' -kill_cnt 60  -seed 2 -noise_rate 0 -neg_num 1  -use_all_neg_samples -lr 0.0001 -batch 512   -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.2  -init_dim 100   -k_w 10 -k_h 20  
   
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'conve' -data 'FB15k-237' -invest_mode 'aggre' -kill_cnt 20  -noise_rate 0 -neg_num 10  -seed 2 -rgcn_num_blocks 100  -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.1 -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'conve' -data 'FB15k-237' -invest_mode 'mlp'  -noise_rate 0  -seed 2 -neg_num 1454  -lr 0.0001 -batch 512   -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  
     ### use all nodes as negative samples
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'conve' -data 'FB15k-237' -invest_mode 'mlp'  -noise_rate 0 -seed 2 -neg_num 10 -use_all_neg_samples -lr 0.0001 -batch 512   -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  
   

   
    #########################rgcn fb15k
    #-no_edge_reverse
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k' -invest_mode 'aggre' -kill_cnt 20  -noise_rate 0 -neg_num 1 -rgcn_num_blocks 100   -lr 0.01 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 600  -gcn_dim 600 -k_w 10 -k_h 60  
    
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k' -invest_mode 'mlp'  -kill_cnt 20  -noise_rate 0 -neg_num 1 -rgcn_num_blocks 100 -lr 0.01 -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 600  -gcn_dim 600  -k_w 10 -k_h 60  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k' -invest_mode 'score'    -noise_rate 0 -neg_num 1 -rgcn_num_blocks 100 -lr 0.0001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.2  -init_dim 600   -k_w 10 -k_h 20  
    
    ####################rgcn wn18
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18' -invest_mode 'aggre' -kill_cnt 20 -noise_rate 0 -neg_num 10 -rgcn_num_blocks 100 -lr 0.00001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.05 -init_dim 100 -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18' -invest_mode 'mlp'  -kill_cnt 20 -noise_rate 0 -neg_num 1 -rgcn_num_blocks 100 -lr 0.0001 -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18' -invest_mode 'score'  -seed 2 -kill_cnt 20 -noise_rate 0 -neg_num 1 -rgcn_num_blocks 100 -lr 0.001 -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.05  -init_dim 100   -k_w 10 -k_h 20  
    
    ####################rgcn wn18rr
    python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18RR' -invest_mode 'aggre'  -seed 2 -noise_rate 0 -neg_num 10 -rgcn_num_blocks 100 -lr 0.0001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.1 -init_dim 100 -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18RR' -invest_mode 'aggre'  -seed 2 -noise_rate 0 -neg_num 1 -use_all_neg_samples -rgcn_num_blocks 100 -lr 0.0001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.3 -init_dim 100 -k_w 10 -k_h 20  
    
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'conve' -data 'WN18RR' -invest_mode 'aggre' -seed 2 -noise_rate 0 -neg_num 10 -rgcn_num_blocks 100 -lr 0.0001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.1 -init_dim 100 -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'conve' -data 'WN18RR' -invest_mode 'aggre'  -noise_rate 0 -neg_num 10 -use_all_neg_samples -rgcn_num_blocks 100 -lr 0.0001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.1 -init_dim 100 -k_w 10 -k_h 20  
    

    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'conve' -data 'WN18RR' -invest_mode 'mlp'  -noise_rate 0 -seed 2 -neg_num 4094 -rgcn_num_blocks 100  -lr 0.0001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.1 -init_dim 100 -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'conve' -data 'WN18RR' -invest_mode 'mlp'  -noise_rate 0 -seed 2 -neg_num 10 -use_all_neg_samples -rgcn_num_blocks 100 -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.3 -init_dim 100 -k_w 10 -k_h 20  
    
   
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18RR' -invest_mode 'mlp'  -noise_rate 0 -seed 2 -neg_num 1 -rgcn_num_blocks 100 -lr 0.0001 -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.1 -init_dim 100 -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18RR' -invest_mode 'mlp'  -seed 2  -noise_rate 0 -neg_num 1 -use_all_neg_samples -rgcn_num_blocks 100 -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.05 -init_dim 100 -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18RR' -invest_mode 'score'  -noise_rate 0 -neg_num 1 -rgcn_num_blocks 100 -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.1 -init_dim 100 -k_w 10 -k_h 20  
    

    ######################### distmult
    # python run.py -data_dir './afs' -output_dir './output'  -model 'distmult' -score_func 'distmult' -data 'FB15k-237' -invest_mode 'score'  -loss_func 'marginRank' -neg_num 1  -noise_rate 0 -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -margin 10. -gamma 0. -hid_drop 0.3  -init_dim 200   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'transe' -score_func 'transe' -data 'FB15k-237' -invest_mode 'score'  -loss_func 'marginRank' -neg_num 1  -noise_rate 0 -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -margin 5. -gamma 0. -hid_drop 0.3  -init_dim 200   -k_w 10 -k_h 20  
    
    ###mlp + marginranking
    # python run.py -data_dir './afs' -output_dir './output'  -model 'distmult' -score_func 'distmult' -data 'FB15k-237' -invest_mode 'mlp'  -loss_func 'marginRank' -neg_num 1  -use_all_neg_samples  -noise_rate 0 -lr 0.001 -batch 512  -l2 0.0 -num_workers 5 -margin 5 -gamma 0. -hid_drop 0.5  -init_dim 100   -k_w 10 -k_h 20 

    # python run.py -data_dir './afs' -output_dir './output'  -model 'distmult' -score_func 'distmult' -data 'FB15k-237' -invest_mode 'mlp'  -loss_func 'marginRank' -neg_num 1   -noise_rate 0 -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -margin 10 -gamma 0. -hid_drop 0.5  -init_dim 100   -k_w 10 -k_h 20  
    

	############################### noise in aggregation

    ###compgcn
	# python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn'  -score_func 'distmult'  -opn 'sub' -data 'FB15k-237' -invest_mode 'aggre' -noise_rate 0.5 -all_noise 0 -lr 0.001  -batch 128 -gcn_layer 2 -l2 0.0 -num_workers 10 -hid_drop 0.3  -init_dim 100 -gcn_dim 150  -k_w 10 -k_h 20  -gamma 40 
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'conve' -opn 'corr' -data 'FB15k-237' -invest_mode 'aggre'  -noise_rate 1 -all_noise 1  -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  
   

	# python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'conve'  -opn 'corr' -data 'WN18RR' -invest_mode 'aggre'  -seed 2 -noise_rate 1. -all_noise 1 -neg_num 0  -lr 0.001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 1  -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult'  -opn 'sub' -data 'WN18' -invest_mode 'aggre'  -noise_rate 1. -all_noise 1. -neg_num 0 -lr 0.001  -batch 512  -l2 0.0 -num_workers 10 -hid_drop 0.2  -init_dim 100   -k_w 10 -k_h 20  
    

    ####rgcn 
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18' -invest_mode 'aggre' -kill_cnt 20 -noise_rate 1 -all_noise 1. -neg_num 10 -rgcn_num_blocks 100 -lr 0.001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237' -invest_mode 'aggre'   -kill_cnt 20  -seed 2 -noise_rate 1 -all_noise 1. -neg_num 10 -rgcn_num_blocks 100 -lr 0.0001  -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18RR' -invest_mode 'aggre' -kill_cnt 20 -seed 2 -noise_rate 1 -all_noise 1. -neg_num 10 -rgcn_num_blocks 100 -lr 0.001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  
    
        ####no reverse in aggre
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237' -invest_mode 'aggre'  -no_edge_reverse  -epoch 100 -kill_cnt 20 -noise_rate 5. -all_noise 1. -neg_num 10 -rgcn_num_blocks 100 -lr 0.001  -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18' -invest_mode 'aggre'   -kill_cnt 20 -noise_rate 5. -all_noise 1. -neg_num 10 -rgcn_num_blocks 100 -lr 0.0001  -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.05  -init_dim 100   -k_w 10 -k_h 20  
    
    ############################### change loss
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237' -invest_mode 'aggre' -change_loss 1.  -seed 2 -noise_rate 0 -neg_num 0 -rgcn_num_blocks 100  -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.2 -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237' -invest_mode 'mlp' -change_loss 1.  -noise_rate 0 -neg_num 0 -rgcn_num_blocks 100 -seed 2 -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.3 -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'conve' -data 'FB15k-237' -invest_mode 'mlp' -change_loss 1.  -noise_rate 0 -neg_num 0 -rgcn_num_blocks 100 -seed 2 -lr 0.0001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.3 -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'conve' -data 'FB15k-237' -invest_mode 'aggre' -change_loss 1.  -noise_rate 0 -neg_num 0 -rgcn_num_blocks 100  -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.3 -init_dim 100   -k_w 10 -k_h 20  
    
    
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18RR' -invest_mode 'aggre' -change_loss 1.  -noise_rate 0 -neg_num 0 -rgcn_num_blocks 100  -seed 2 -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.3 -init_dim 100   -k_w 10 -k_h 20  
    
    

    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'WN18RR' -invest_mode 'mlp' -change_loss 1. -seed 2 -noise_rate 0 -neg_num 0 -rgcn_num_blocks 100  -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.1 -init_dim 100   -k_w 10 -k_h 20  
    
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'conve' -data 'WN18RR' -invest_mode 'mlp' -change_loss 1. -seed 2 -noise_rate 0 -neg_num 0 -rgcn_num_blocks 100  -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.3 -init_dim 100   -k_w 10 -k_h 20  
    
    

    ##########compgcn   fb15k

    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult' -opn 'sub' -data 'FB15k-237' -invest_mode 'aggre' -change_loss 1.  -noise_rate 0 -neg_num 10  -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.2 -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult' -opn 'sub' -data 'FB15k-237' -invest_mode 'aggre' -change_loss 1.  -noise_rate 0 -neg_num 10  -use_all_neg_samples -lr 0.0001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.2 -init_dim 100   -k_w 10 -k_h 20  

    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult' -opn 'sub' -data 'FB15k-237' -invest_mode 'mlp' -change_loss 1. -seed 2  -noise_rate 0 -neg_num 10  -lr 0.0001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.2 -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult' -opn 'sub' -data 'FB15k-237' -invest_mode 'mlp' -change_loss 1.  -seed 2 -noise_rate 0 -neg_num 1 -use_all_neg_samples -lr 0.0001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.2 -init_dim 100   -k_w 10 -k_h 20  
    

    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'conve' -opn 'sub' -data 'FB15k-237' -invest_mode 'mlp' -change_loss 1. -noise_rate 0 -neg_num 50  -lr 0.0001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.2 -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'conve' -opn 'corr' -data 'FB15k-237' -invest_mode 'aggre' -change_loss 1. -seed 2 -noise_rate 0 -neg_num 7270  -lr 0.0001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.1 -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'conve' -opn 'corr' -data 'FB15k-237' -invest_mode 'aggre' -change_loss 1.  -noise_rate 0 -neg_num 200  -use_all_neg_samples -lr 0.0001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.1 -init_dim 100   -k_w 10 -k_h 20  
    
    
    
    ####wn18rr

    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult' -opn 'sub' -data 'WN18RR' -invest_mode 'aggre' -change_loss 1.  -noise_rate 0 -neg_num 10  -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.3 -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult' -opn 'sub' -data 'WN18RR' -invest_mode 'aggre' -change_loss 1. -seed 2 -noise_rate 0 -neg_num 10 -use_all_neg_samples -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.3 -init_dim 100   -k_w 10 -k_h 20  
    
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'conve' -opn 'corr' -data 'WN18RR' -invest_mode 'aggre' -change_loss 1.  -noise_rate 0 -neg_num 20472  -seed 2 -lr 0.01 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.1 -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'conve' -opn 'corr' -data 'WN18RR' -invest_mode 'aggre' -change_loss 1.  -seed 2 -noise_rate 0 -neg_num 10 -use_all_neg_samples -lr 0.01 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.1 -init_dim 100   -k_w 10 -k_h 20  
    
    
   

    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult' -opn 'sub' -data 'WN18RR' -invest_mode 'mlp' -change_loss 1.   -noise_rate 0 -neg_num 10  -lr 0.01 -batch 512  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.05 -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult' -opn 'sub' -data 'WN18RR' -invest_mode 'mlp' -change_loss 1.  -noise_rate 0 -neg_num 10  -use_all_neg_samples -lr 0.001 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.3 -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'conve' -opn 'sub' -data 'WN18RR' -invest_mode 'mlp' -change_loss 1.  -noise_rate 0 -neg_num 10  -lr 0.01 -batch 512  -l2 0.0 -num_workers 10 -gcn_layer 1 -hid_drop 0.3 -init_dim 100   -k_w 10 -k_h 20  
    

    ############################### noise in loss
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn'  -score_func 'distmult' -opn 'sub' -data 'FB15k-237' -invest_mode 'aggre' -loss_noise_rate 5 -all_loss_noise 1 -lr 0.001  -batch 512 -gcn_layer 2 -l2 0.0 -num_workers 10 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20 
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn'  -score_func 'distmult' -opn 'sub' -data 'FB15k-237' -invest_mode 'mlp' -loss_noise_rate 1 -all_loss_noise 1 -lr 0.001  -batch 512 -gcn_layer 2 -l2 0.0 -num_workers 10 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20 

    ####operator!!!!
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn'  -score_func 'conve' -opn 'corr' -data 'WN18RR' -invest_mode 'aggre' -loss_noise_rate 1. -lr 0.001  -batch 512 -gcn_layer 2 -l2 0.0 -num_workers 10 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20 
    
    #########stronger noise:
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn'  -score_func 'distmult' -opn 'sub' -data 'FB15k-237' -invest_mode 'aggre' -loss_noise_rate 5 -all_loss_noise 0. -strong_noise -lr 0.001  -batch 512 -gcn_layer 2 -l2 0.0 -num_workers 10 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20 
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn'  -score_func 'distmult' -opn 'sub' -data 'FB15k-237' -invest_mode 'mlp' -loss_noise_rate 1 -all_loss_noise 1. -strong_noise -lr 0.001  -batch 512 -gcn_layer 2 -l2 0.0 -num_workers 10 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20 
    
    ###########rgcn
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237' -invest_mode 'aggre'  -kill_cnt 20  -loss_noise_rate 2. -neg_num 1 -rgcn_num_blocks 100 -lr 0.001  -batch 1024  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  
    
    ################################ remove triplets in loss
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn'  -score_func 'distmult' -opn 'sub' -data 'FB15k-237' -invest_mode 'aggre'  -seed 2 -left_loss_tri_rate 0.05 -lr 0.001  -batch 512 -gcn_layer 2 -l2 0. -num_workers 10 -hid_drop 0.4  -init_dim 100   -k_w 10 -k_h 20 
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn'  -score_func 'distmult' -opn 'sub' -data 'FB15k-237' -invest_mode 'mlp'  -seed 2 -left_loss_tri_rate 0.05 -lr 0.001  -batch 512 -gcn_layer 2 -l2 0. -num_workers 10 -hid_drop 0.4  -init_dim 100   -k_w 10 -k_h 20 
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn'  -score_func 'distmult' -opn 'sub' -data 'FB15k-237' -invest_mode 'score' -seed 2 -left_loss_tri_rate 0.05 -lr 0.001  -batch 512 -gcn_layer 2 -l2 0. -num_workers 10 -hid_drop 0.2  -init_dim 100   -k_w 10 -k_h 20 
    ##### remove triplets in loss and aggregation
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn'  -score_func 'distmult' -opn 'sub' -data 'FB15k-237' -invest_mode 'aggre'  -less_edges_in_aggre  -left_loss_tri_rate 0.05 -lr 0.0001  -batch 512 -gcn_layer 2 -l2 0. -num_workers 10 -hid_drop 0.5  -init_dim 100   -k_w 10 -k_h 20 
    


    ############################ noise triplets + true triplets
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn'  -score_func 'distmult' -opn 'sub' -data 'FB15k-237' -invest_mode 'aggre' -add_triplet_rate 1. -add_triplet_base_noise 5 -lr 0.001  -batch 512 -gcn_layer 2 -l2 0.0 -num_workers 10 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20 
    

    #####################remove loss + noise in aggregation
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn'  -score_func 'distmult' -opn 'sub' -data 'FB15k-237' -invest_mode 'aggre'  -noise_rate 1 -all_noise 1 -left_loss_tri_rate 0.05 -lr 0.001  -batch 512 -gcn_layer 2 -l2 0. -num_workers 10 -hid_drop 0.4  -init_dim 100   -k_w 10 -k_h 20 


    ##################### use input feature in the aggregation
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult' -opn 'sub' -data 'FB15k-237' -invest_mode 'aggre' -neg_num 0 -noise_rate 0 -use_feat_input  -lr 0.0001 -batch 128  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.6  -init_dim 200   -k_w 10 -k_h 20 
      
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult' -opn 'sub' -data 'FB15k-237' -invest_mode 'mlp' -neg_num 0 -noise_rate 0 -use_feat_input  -lr 0.0001 -batch 128  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.6  -init_dim 200   -k_w 10 -k_h 20 
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult' -opn 'sub' -data 'FB15k-237' -invest_mode 'score' -neg_num 0 -noise_rate 0 -use_feat_input  -lr 0.001 -batch 128  -l2 0. -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 200   -k_w 10 -k_h 20 
      
    ############################## reduce triplets in train
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult' -opn 'sub' -data 'FB15k-237/small_train_split' -invest_mode 'aggre' -neg_num 0 -noise_rate 0 -lr 0.00001 -batch 128  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.1  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult' -opn 'sub' -data 'FB15k-237/small_train_split' -invest_mode 'mlp' -neg_num 0 -noise_rate 0 -lr 0.0001 -batch 128  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.6  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult' -opn 'sub' -data 'FB15k-237/small_train_split' -invest_mode 'score' -neg_num 0 -noise_rate 0 -lr 0.0001 -batch 128  -l2 0.0 -num_workers 10 -gcn_layer 2 -hid_drop 0.6  -init_dim 100   -k_w 10 -k_h 20  
   
      ####### use noise
    # python run.py -data_dir './afs' -output_dir './output'  -model 'compgcn' -score_func 'distmult' -opn 'sub' -data 'FB15k-237' -invest_mode 'aggre' -neg_num 0 -noise_rate 1. -all_noise 1. -use_feat_input  -lr 0.001 -batch 128  -l2 1e-8 -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 200   -k_w 10 -k_h 20 
    
    ############################### bpr loss
    ####use all negative samples
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237' -invest_mode 'mlp'  -loss_func 'bpr' -noise_rate 0 -neg_num 1  -use_all_neg_samples -lr 0.0001 -batch 512   -l2 1e-4 -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  
    
    # python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237' -invest_mode 'mlp'  -loss_func 'bpr'  -noise_rate 0 -neg_num 1  -lr 0.001 -batch 512   -l2 1e-8 -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  
   
   ################################ log softmax loss
#    python run.py -data_dir './afs' -output_dir './output'  -model 'rgcn' -score_func 'distmult' -data 'FB15k-237' -invest_mode 'mlp'  -loss_func 'logsoftmax' -noise_rate 0 -neg_num 1   -lr 0.0001 -batch 512   -l2 1e-8 -num_workers 10 -gcn_layer 2 -hid_drop 0.3  -init_dim 100   -k_w 10 -k_h 20  
    

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
