import os



# ###### com: conve+WN18RR : aggre, mlp, score

lrs = [0.001]
l2 = '0'
hid_drops = [0.3]

for lr in lrs:
    lr = str(lr)
    for hid_drop in hid_drops:

        hid_drop = str(hid_drop)
        

        command2run = 'sbatch job.sb ' + lr + ' '  + hid_drop + ' '+ l2
        os.system(command2run)



###### rgcn: conve+WN18RR : aggre, mlp, score
# model ='rgcn'
# score_func = 'distmult'
# opn = 'sub'
# invest_modes =  ['mlp']
# neg_num = '10'
# noise_rate = '0'
# lrs = [0.001,0.0001]
# batch = '512'
# l2 = '0'
# gcn_layer = '2'
# hid_drops = [0.05,0.1, 0.3,0.5]
# init_dim = '100'
# k_w = '10'
# k_h = '20'
# data = 'WN18RR'
# seed = '41504'
# kill_cnt = '20'
#
# for lr in lrs:
#     lr = str(lr)
#     for hid_drop in hid_drops:
#
#         hid_drop = str(hid_drop)
#         for invest_mode in invest_modes:
#
#             invest_mode = str(invest_mode)
#             command2run = 'sbatch job.sb ' + model + ' '  + score_func + ' ' + opn + ' ' + data + ' ' + invest_mode + ' ' + neg_num + ' ' + noise_rate + ' ' + lr + ' ' + batch + ' '+l2+' ' + gcn_layer + ' ' + hid_drop + ' '+init_dim + ' ' + k_w + ' ' + k_h + ' ' + seed + ' ' + kill_cnt
#             os.system(command2run)

######################traditional distmult
# model ='distmult'
# score_func = 'distmult'
# opn = 'sub'
# data = 'WN18'
# invest_mode = 'score'
# loss_func = 'marginRank'
# neg_num = '10'
# noise_rate = '0'
# lrs = [0.001,0.0001]
# batch = '512'
# l2 = '0'
# margins = [5,10,30]
# hid_drops = [0.05, 0.1, 0.3, 0.5]
# init_dim = '200'
# k_w = '10'
# k_h = '20'
# gcn_layer = '1'
# kill_cnt = '60'
# seed = '41504'
# 
# for lr in lrs:
#     lr = str(lr)
#     for hid_drop in hid_drops:
# 
#         hid_drop = str(hid_drop)
#         for margin in margins:
# 
#             margin = str(margin)
#             command2run = 'sbatch job.sb ' + model + ' '  + score_func + ' ' + opn + ' ' + data + ' ' + invest_mode + ' ' + neg_num + ' ' + noise_rate + ' ' + lr + ' ' + batch + ' '+l2+' ' + gcn_layer + ' ' + hid_drop + ' '+init_dim + ' ' + k_w + ' ' + k_h + ' ' + seed +  ' '+margin + ' ' + loss_func
#             os.system(command2run)