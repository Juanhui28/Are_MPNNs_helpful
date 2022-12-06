# %%
from helper import *
# from read_data.data_loader import *
from read_data.read_data_compgcn import read_compgcn
from read_data.read_data_rgcn import read_rgcn
import gzip
import random
import math

# sys.path.append('./')
from model.models import *
from model.rgcn_model import *



class Runner(object):

    def __init__(self, params):
        
        self.p			= params
        self.logger		= get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

        
        pprint(vars(self.p))
        
        self.n_gpu = torch.cuda.device_count()
        
        if self.p.gpu != '-1' and torch.cuda.is_available():

            self.device = torch.device('cuda:0')
            
            self.n_gpu=1

            torch.cuda.manual_seed(self.p.seed)
            torch.cuda.manual_seed_all(self.p.seed)
            
            
            torch.backends.cudnn.deterministic = True
            
        else:
            self.device = torch.device('cpu')

        
        self.model        = self.add_model(self.p.model, self.p.score_func)
        self.optimizer    = self.add_optimizer(self.model.parameters())



    def add_model(self, model, score_func):
        
        if self.p.read_setting == 'no_negative_sampling': self.read_data = read_compgcn(self.p)
        elif self.p.read_setting == 'negative_sampling': 
            if model == 'transe':
                self.read_data = read_rgcn(self.p, triplet_reverse_loss=True)
            else:
                self.read_data = read_rgcn(self.p)
        
        
        else: raise NotImplementedError('please choose one reaing setting: no_negative_sampling or negative_sampling')

        edge_index, edge_type, self.data_iter, self.feature_embeddings, indices_2hop = self.read_data.load_data()

        

        print('################### model:'+ self.p.model + ' #################')
        print('reading setting: ', self.p.read_setting)

        if  self.p.read_setting == 'no_negative_sampling': 
            if self.p.neg_num != 0: raise ValueError('no negative sampling does not use negative sampling, please the predefined parameter ''neg_num'' be 0')
            print('no negative samples: ', self.p.neg_num)

        elif self.p.read_setting == 'negative_sampling': 
            if self.p.neg_num <= 0: raise ValueError('use negative sampling, please the predefined parameter ''neg_num'' to be larger than 0')
            
            if self.p.use_all_neg_samples:
                print('use all possible negative samples ')
            else:
                print('negative samples: ', self.p.neg_num)
        

        model = BaseModel(edge_index, edge_type, params=self.p, feature_embeddings=self.feature_embeddings, indices_2hop=indices_2hop)

        model.to(self.device)
        if self.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        
        return model

    def add_optimizer(self, parameters):
        
        return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)



    def save_model(self, save_path):
        
        state = {
            'state_dict'	: self.model.state_dict(),
            'best_val'	: self.best_val,
            'best_epoch'	: self.best_epoch,
            'optimizer'	: self.optimizer.state_dict(),
            'args'		: vars(self.p)
        }
        torch.save(state, save_path)

    def load_model(self, load_path):
        
        state			= torch.load(load_path)
        state_dict		= state['state_dict']
        param  = state['args']
        self.best_val		= state['best_val']
        self.best_val_mrr	= self.best_val['mrr'] 

        state_dict_copy = dict()
        for k in state_dict.keys():
            
            state_dict_copy[k] = state_dict[k]
        
        print('load params: ', param)
        self.model.load_state_dict(state_dict_copy)
        

    def evaluate(self, split, epoch, mode, f_test):
        
        
        # left_results  = self.predict(split=split, mode='tail_batch')
        # right_results = self.predict(split=split, mode='head_batch')

        left_results  = self.ensemble(split=split, mode='tail_batch')
        right_results = self.ensemble(split=split, mode='head_batch')


        results       = get_combined_results(left_results, right_results)
        self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))
        if mode == 'val':
            self.file_val_result.write(str(results['mrr'])+'\n')
            self.file_val_result.flush()
        if mode == 'test':
            self.file_test_result.write(str(results['mrr'])+'\n')
            self.file_test_result.flush()
            
        if mode == 'test' and self.best_update == 1:
            f_test.write('left right MRR: '+str(results['left_mrr']) + '\t' +str(results['right_mrr']) + '\t' + str(results['mrr']) + '\n' )
            f_test.write('left right MR: '+str(results['left_mr']) + '\t' +str(results['right_mr']) + '\t' + str(results['mr']) + '\n' )
            f_test.write('left right hits@1: '+str(results['left_hits@1']) + '\t' +str(results['right_hits@1']) + '\t' + str(results['hits@1']) + '\n' )
            f_test.write('left right hits@3: '+str(results['left_hits@3']) + '\t' +str(results['right_hits@3']) + '\t' + str(results['hits@3']) + '\n' )
            f_test.write('left right hits@10: '+str(results['left_hits@10']) + '\t' +str(results['right_hits@10']) + '\t' + str(results['hits@10']) + '\n' )
            f_test.write('****************************************************\n')
            f_test.flush()
            self.best_update = 0

        return results

    def ensemble(self, split='valid', mode='tail_batch'):
        self.model.eval()

        for i in range(9):
            
            pred_score = torch.load('bin_doc/'+self.p.dataset+ '/'+str(self.p.seed)+'/'+ split+'_'+mode.split('_')[0]+'_'+str(i))
            
            # if i == 5:
            # 	continue
            
            preds = preds + pred_score
            # print('x: ', preds.size())

        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])
            
            for step, batch in enumerate(train_iter):
                sub, rel, obj, label	= self.read_data.read_batch(batch, split, self.device)
                # print(sub[:3], rel[:3], obj[:3])
                pred  = preds[step*self.p.batch_size: (step+1)*self.p.batch_size]
                # print('sub', sub)

                b_range			= torch.arange(pred.size()[0], device=self.device)
                target_pred		= pred[b_range, obj]
                pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] 	= target_pred
                ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]

                ranks 			= ranks.float()
                results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
                results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
                results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
                for k in range(10):
                    results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

            
        return results

    def predict(self, split='valid', mode='tail_batch'):
        
        self.model.eval()
        
        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

            for step, batch in enumerate(train_iter):
                sub, rel, obj, label	= self.read_data.read_batch(batch, split, self.device)

                x, r			= self.model.forward()
            
                pred, _	 = self.model.get_loss(x, r, sub, rel, label, pos_neg_ent=None)
                # torch.save(state, save_path)
                b_range			= torch.arange(pred.size()[0], device=self.device)
                target_pred		= pred[b_range, obj]
                pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] 	= target_pred
                
                ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
                
                ranks 			= ranks.float()
                results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
                results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
                results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
                for k in range(10):
                    results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

                if step == 0:
                    all_pred = pred
                    print(all_pred.size())
                else:
                    all_pred = torch.cat([all_pred, pred], dim=0)
            
            # torch.save(all_pred, 'bin_doc/'+self.p.dataset+ '/'+str(self.p.seed)+'/'+ split+'_'+mode.split('_')[0]+'_'+str(self.p.index))


        return results


    def run_epoch(self, epoch, val_mrr = 0):
        
        self.model.train()
        losses = []
        train_iter = iter(self.data_iter['train'])
        
        results = {}
        count = 0
        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            
            sub, rel, obj, label, pos_neg_ent = self.read_data.read_batch(batch, 'train', self.device)
            

            x, r	= self.model.forward()
            pred, loss = self.model.get_loss(x, r, sub, rel, label, pos_neg_ent)
            
            if self.n_gpu > 0:
                loss = loss.mean()
            
            
            
            loss.backward()
            
            self.optimizer.step()
            losses.append(loss.item())

            if step % 500 == 0:
                # count   = float(results['count'])
                # ave_train_mrr = round(results ['mrr'] /count, 5)
                ave_train_mrr = 0.0

                self.logger.info('[E:{}| {}]: Train Loss:{:.5}, Train MRR:{:.5}, Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses), ave_train_mrr, self.best_val_mrr, self.p.name))

        loss = np.mean(losses)

        #####################


        self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
        return loss


    def fit(self):
        
        self.best_val_mrr, self.best_val, self.best_test, self.best_epoch, val_mrr = 0., {}, {},0, 0.
        save_path = os.path.join(self.p.output_dir, self.p.name+str(self.p.lr)+'_'+str(self.p.hid_drop)+'_'+str(self.p.l2)+'_'+str(self.p.seed))

        ###########debug

        kill_cnt = 0
        f_test = open(os.path.join(self.p.output_dir, 'mrr_best_scores_'+str(self.p.lr)+'_'+str(self.p.hid_drop)+'_'+str(self.p.l2)+'_'+str(self.p.seed)+'.txt'), 'w')
        f_train_result =  open(os.path.join(self.p.output_dir, 'mrr_train_scores'+str(self.p.lr)+'_'+str(self.p.hid_drop)+'_'+str(self.p.l2)+'_'+str(self.p.seed)+'.txt'), 'w')
        f_val_result = open(os.path.join(self.p.output_dir, 'mrr_val_scores'+str(self.p.lr)+'_'+str(self.p.hid_drop)+'_'+str(self.p.l2)+'_'+str(self.p.seed)+'.txt'), 'w')
        f_test_result = open(os.path.join(self.p.output_dir, 'mrr_test_scores'+str(self.p.lr)+'_'+str(self.p.hid_drop)+'_'+str(self.p.l2)+'_'+str(self.p.seed)+'.txt'), 'w')

        

        self.file_train_result = f_train_result
        self.file_val_result = f_val_result
        self.file_test_result = f_test_result

        for epoch in range(self.p.max_epochs):
            train_loss  = self.run_epoch(epoch, val_mrr)  

            if epoch % self.p.evaluate_every == 0: 
                self.best_update = 0
                val_results = self.evaluate('valid', epoch, 'val',f_test)

                if val_results['mrr'] > self.best_val_mrr:
                    self.best_update = 1
                    self.best_val	   = val_results
                    self.best_val_mrr  = val_results['mrr']
                    self.best_epoch	   = epoch
                    self.save_model(save_path)   
                    kill_cnt = 0
            

                else:
                    kill_cnt += 1
                    if kill_cnt % 10 == 0 and self.p.gamma > 5:
                        self.p.gamma -= 5 
                        self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
                    if kill_cnt > self.p.kill_cnt: 
                        self.logger.info("Early Stopping!!")
                        break
                
                self.logger.info('Evaluating on Test data')

                test_results = self.evaluate('test', epoch, 'test', f_test)

                
                self.logger.info('[Epoch {}]: Training Loss: {:.5}, Valid MRR: {:.5}\n\n'.format(epoch, train_loss, self.best_val_mrr)) #debug

        
        
    # %%		
if __name__ == '__main__':
    # %%	
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-name',		default='testrun',					help='Set run name for saving/restoring models')
    ######################## compgcn
    parser.add_argument('-data',		dest='dataset',         default='FB15k-237',            help='Dataset to use, default: FB15k-237')
    parser.add_argument('-model',		dest='model',		default='compgcn',		help='Model Name')
    parser.add_argument('-score_func',	dest='score_func',	default='distmult',		help='Score Function for Link prediction')
    parser.add_argument('-opn',             dest='opn',             default='sub',                 help='Composition Operation to be used in CompGCN')
    parser.add_argument('-loss_func',	dest='loss_func',	default='bce',		help='Loss Function for Link prediction')


    parser.add_argument('-batch',           dest='batch_size',      default=128,    type=int,       help='Batch size')
    parser.add_argument('-kill_cnt',           dest='kill_cnt',      default=60,    type=int,       help='early stopping')
    parser.add_argument("-evaluate_every", type=int, default=1,  help="perform evaluation every n epochs")
    parser.add_argument('-gamma',		type=float,             default=40.0,			help='Margin in the transe score')
    parser.add_argument('-gpu',		type=str,               default='0',			help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-epoch',		dest='max_epochs', 	type=int,       default=9999999,  	help='Number of epochs')
    parser.add_argument('-l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('-lr',		type=float,             default=0.001,			help='Starting Learning Rate')
    parser.add_argument('-lbl_smooth',      dest='lbl_smooth',	type=float,     default=0.1,	help='Label Smoothing')
    parser.add_argument('-num_workers',	type=int,               default=0,                     help='Number of processes to construct batches')
    parser.add_argument('-seed',            dest='seed',            default=41504,  type=int,     	help='Seed for randomization')

    parser.add_argument('-restore',         dest='restore',         action='store_true',   default=False,         help='Restore from the previously saved model')
    parser.add_argument('-bias',            dest='bias',            action='store_true',            help='Whether to use bias in the model')

    parser.add_argument('-compgcn_num_bases',	dest='compgcn_num_bases', 	default=-1,   	type=int, 	help='Number of basis relation vectors to use')
    parser.add_argument('-init_dim',	dest='init_dim',	default=100,	type=int,	help='Initial dimension size for entities and relations')
    parser.add_argument('-gcn_dim',	  	dest='gcn_dim', 	default=200,   	type=int, 	help='Number of hidden units in GCN')
    parser.add_argument('-embed_dim',	dest='embed_dim', 	default=None,   type=int, 	help='Embedding dimension to give as input to score function')
    parser.add_argument('-gcn_layer',	dest='gcn_layer', 	default=1,   	type=int, 	help='Number of GCN Layers to use')
    parser.add_argument('-gcn_drop',	dest='dropout', 	default=0.1,  	type=float,	help='Dropout to use in GCN Layer')
    parser.add_argument('-hid_drop',  	dest='hid_drop', 	default=0.2,  	type=float,	help='Dropout after GCN')


    # ConvE specific hyperparameters
    parser.add_argument('-hid_drop2',  	dest='hid_drop2', 	default=0.3,  	type=float,	help='ConvE: Hidden dropout')
    parser.add_argument('-feat_drop', 	dest='feat_drop', 	default=0.3,  	type=float,	help='ConvE: Feature Dropout')
    parser.add_argument('-k_w',	  	dest='k_w', 		default=10,   	type=int, 	help='ConvE: k_w')
    parser.add_argument('-k_h',	  	dest='k_h', 		default=20,   	type=int, 	help='ConvE: k_h')
    parser.add_argument('-num_filt',  	dest='num_filt', 	default=200,   	type=int, 	help='ConvE: Number of filters in convolution')
    parser.add_argument('-ker_sz',    	dest='ker_sz', 		default=7,   	type=int, 	help='ConvE: Kernel size to use')


    ###### rgcn
    parser.add_argument('-neg_num',	  	dest='neg_num', 	default=0,   	type=int, 	help='Number of negative samples')
    parser.add_argument('-rgcn_num_bases',	dest='rgcn_num_bases', 	default=None,   	type=int, 	help='Number of basis relation vectors to use in rgcn')
    parser.add_argument('-rgcn_num_blocks',	dest='rgcn_num_blocks', 	default=100,   	type=int, 	help='Number of block relation vectors to use in rgcn layer1')
    parser.add_argument('-no_edge_reverse',	dest='no_edge_reverse', 	action='store_true',   default=False,   	help='whether to use the reverse relation in the aggregation')
    ### use all possible negative samples
    parser.add_argument('-use_all_neg_samples',	dest='use_all_neg_samples', 	action='store_true',   default=False,   	help='whether to use the ')

    ####### margin loss
    parser.add_argument('-margin',		type=float,             default=10.0,			help='Margin in the marginRankingLoss')

    ############ config
    parser.add_argument("-data_dir", default='./data',type=str,required=False, help="The input data dir.")
    parser.add_argument("-output_dir", default='./output_test',type=str,required=False, help="The input data dir.")
    parser.add_argument('-logdir',          dest='log_dir',         default='./log/',               help='Log directory')
    parser.add_argument('-config',          dest='config_dir',      default='./config/',            help='Config directory')

    ###adding noise in aggregation
    parser.add_argument("-noise_rate", type=float, default=0., help="the rate of noise edges adding  in aggregation, but not loss")
    parser.add_argument("-all_noise", type=float, default=0., help="use noises to edges in aggregation, 1: only use noise edges, 0: add noise edges")


    #####  noise in loss
    parser.add_argument("-loss_noise_rate", type=float, default=0, help="true triplets +  adding noise in loss")
    parser.add_argument("-all_loss_noise", type=float, default=0., help="use noises to triplets in loss, 1: only use noise triplets, 0: add noise triplets")
    parser.add_argument("-strong_noise", action='store_true',   default=False, help="use the stronger noise or not")



    parser.add_argument("-add_triplet_rate", type=float, default=0., help="noise triplets + adding true triplets in loss: the true triplets rate")
    parser.add_argument("-add_triplet_base_noise", type=float, default=0., help="noise triplets + adding true triplets in loss: the noise rate")


    parser.add_argument("-left_loss_tri_rate", type=float, default=0, help="removing triplets in loss, the left ratio of true triplest in the loss")
    parser.add_argument("-less_edges_in_aggre", action='store_true',   default=False, help="use less triplets in the aggregation (with the same less triplets in the loss)")

    #####  kbgat
    parser.add_argument("-use_feat_input", action='store_true',   default=False,   help="use the node feature as input")

    parser.add_argument("-triplet_no_reverse", 	action='store_true',   default=False,   	help='whether to use another vector to denote the reverse relation in the loss')
    parser.add_argument("-no_partial_2hop", 	action='store_true', default=False)
    parser.add_argument("-alpha",  type=float, default=0.2, help="LeakyRelu alphs for SpGAT layer")
    parser.add_argument("-nheads", type=int,  default=2	, help="Multihead attention SpGAT")
    ####
    parser.add_argument("-read_setting", default='no_negative_sampling',type=str,required=False, help="different reading setting: no_negative_sampling or negative_sampling")


    args = parser.parse_args()


    if not args.restore: args.name = args.name




    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # %%
    model = Runner(args)
    # %%
    model.fit()
