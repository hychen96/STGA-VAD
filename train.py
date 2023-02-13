import sys

import numpy as np
import torch
from torch.autograd import Variable
from utils.utils import *
from tqdm import tqdm
from utils.eval_utils import eval, cal_false_alarm
import argparse

from model import STGA

from adj_cal import *
from torch.utils.data.dataloader import DataLoader
from dataset import Train_Dataset, Test_Dataset
# from dataset_self import Train_Dataset, Test_Dataset
import torch.nn.functional as F
import h5py


def parser_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='I3D_RGB')    #I3D_RGB/C3D_RGB
    parser.add_argument('--dataset', type=str, default='tad')

    parser.add_argument('--size', type=int, default=1024)    # I3D_RGB:1024, C3D_RGB:4096  ucf_crime:2048
    parser.add_argument('--segment_len', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--topk', type=int,default=7)
    # part num should consider the average len of the video
    parser.add_argument('--part_num', type=int, default=32)
    parser.add_argument('--part_len', type=int, default=7)

    parser.add_argument('--epochs', type=int, default=601)
    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('-optimizer', type=str, default='adagrad')

    parser.add_argument('--dropout_rate', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample', type=str, default='random', help='[random/uniform]')

    parser.add_argument('--norm', type=int, default=2)
    parser.add_argument('--clip', type=float, default=4.0)

    parser.add_argument('--lambda_1', type=float, default=0.01)
    parser.add_argument('--lambda_2', type=float, default=0)

    parser.add_argument('--machine', type=str,default='./data/')

    parser.add_argument('--feature_rgb_path', type=str, default='SHT_i3d_feature_rgb.h5')
    parser.add_argument('--model_path_log', type=str, default='log/')
    parser.add_argument('--training_txt', type=str, default='SH_Train.txt')
    parser.add_argument('--testing_txt', type=str, default='SHT_Test.txt')
    parser.add_argument('--test_mask_dir', type=str, default='test_frame_mask/')

    parser.add_argument('--smooth_len', type=int, default=5)

    args = parser.parse_args()
    
    args.feature_rgb_path = args.machine+args.feature_rgb_path
    args.model_path_log = args.machine+args.model_path_log
    args.training_txt = args.machine+args.training_txt
    args.testing_txt = args.machine+args.testing_txt
    args.test_mask_dir = args.machine+args.test_mask_dir


    return args


feature_type = {
    'I3D_RGB': [16, 1024],
    'C3D_RGB': [16, 4096],
}

# bceloss = torch.nn.BCELoss(reduction='mean')

def topk_rank_loss(args, y_pred):
    topk_pred = torch.mean(
        torch.topk(y_pred.view([args.batch_size*2, args.part_num*args.part_len]), args.topk, dim=-1)[0], dim=-1, keepdim=False)
    #y_pred=(80,224=32*7,1) topk_pred=(80,)

    nor_max = topk_pred[:args.batch_size]
    abn_max = topk_pred[args.batch_size:]

    # nor_loss = bceloss(nor_max, torch.zeros(args.batch_size).cuda())

    err = 0
    for i in range(args.batch_size):
        err += torch.sum(F.relu(1-abn_max+nor_max[i]))
    err = err/args.batch_size**2

    abn_pred = y_pred[args.batch_size:]
    spar_l1 = torch.mean(abn_pred)
    nor_pred = y_pred[:args.batch_size]
    smooth_l2 = torch.mean((nor_pred[:, :-1]-nor_pred[:, 1:])**2)

    loss = err+args.lambda_1*spar_l1

    return loss, err, spar_l1, smooth_l2



def train(args):

    dataset = Train_Dataset(args.part_num, args.part_len, args.feature_rgb_path, args.training_txt, args.sample, None, args.norm)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = STGA(nfeat=args.size, nclass=1).cuda().train()
    # union = Union(model).cuda().train()

    # optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    optimizer = torch.optim.RMSprop(model.parameters(),
                    lr=0.001,
                    alpha=0.99,
                    eps=1e-08,
                    weight_decay=5e-4,
                    momentum=0,
                    centered=False)


    test_feats, test_labels, test_annos = Test_Dataset(args.testing_txt, args.test_mask_dir, args.feature_rgb_path, args.norm)
    
    best_AUC = 0
    best_far = 1
    best_AP = 0
    best_iter = 0
    best_far_inter=0
    best_AP_inter=0
    count = 0

    a1_test = []
    for k in range(len(test_feats)):
        dim = test_feats[k].shape[0]
        a = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(i, dim):
                if i == j:
                    a[i][j] = 1.0
                else:
                    a[i][j] = a[j][i] = 1.0 / abs(i - j)
        a1_test.append(a)


    a1 = np.zeros((args.batch_size*2*args.part_num*args.part_len, args.batch_size*2*args.part_num*args.part_len))
    for i in range(args.batch_size*2*args.part_num*args.part_len):
        for j in range(i, args.batch_size*2*args.part_num*args.part_len):
            if i == j:
                a1[i][j] = 1.0
            else:
                a1[i][j] = a1[j][i] = 1.0 / abs(i - j)
    a1 = torch.from_numpy(a1).cuda().float()

    for epoch in range(args.epochs):
        for norm_feats, norm_labs, abnorm_feats, abnorm_labs in dataloader:
            feats = torch.cat([norm_feats, abnorm_feats], dim=0).cuda().float().view([args.batch_size*2, args.part_num*args.part_len, args.size])
            #feats=(80*224*4096)
            # labs = torch.cat([norm_labs, abnorm_labs], dim=0).cuda().float().view(
            #     [args.batch_size * 2, args.part_num * args.part_len, 1])

            feats = feats.view([-1, feats.shape[-1]])

            outputs = model(feats, a1)

            #outputs=(8960*1)
            outputs = outputs.view([args.batch_size * 2, args.part_num * args.part_len, -1])
            #outputs=(80*224*1)
            outputs_mean = torch.mean(outputs, dim=-1, keepdim=True)

            loss, err, l1, l3 = topk_rank_loss(args, outputs_mean)

            optimizer.zero_grad()
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            # if count % 10 == 0:
            print('[{}/{}]: loss {:.4f}, err {:.4f}, l1 {:.4f}, l3 {:.4f}'.format(
                count, epoch, loss, err, l1, l3))
            count += 1
        count = 0
        dataloader.dataset.shuffle_keys()

        if epoch % 5 == 0:

            total_scores = []
            total_labels = []
            total_normal_scores = []

            with torch.no_grad():
                model = model.eval()
                n = 0
                for test_feat, label, test_anno in zip(test_feats, test_labels, test_annos):  #test_feat=47*4096,label='Normal',test_anno=(764,)
                    test_feat = np.array(test_feat).reshape([-1, args.size])  #47*4096
                    temp_score = []

                    a11_test = torch.from_numpy(a1_test[n]).cuda().float()
                    n += 1
                    test_feat = torch.from_numpy(test_feat).cuda().float()
                    # for i in range(test_feat.shape[0]):
                    #     feat = test_feat[i]  #feat=(4096,)
                    #     feat = torch.from_numpy(np.array(feat)).float().cuda().view([-1, args.size])  #feat=(1*4096)
                    #
                    #     # feat = feat.cpu().numpy()
                    #     # f, adj = graph_generator(feat)
                    #     # f = torch.from_numpy(f).cuda().float()
                    #     # adj = torch.from_numpy(adj).cuda().float()
                    #     a1_test = torch.from_numpy(np.array([[1.0]])).cuda().float()
                    #     logits = model(feat, a1_test)
                    #     score = torch.mean(logits).item()
                    #     temp_score.extend([score]*args.segment_len)
                    logits = model(test_feat, a11_test)

                    for i in range(logits.shape[0]):
                        temp_score.extend([logits[i][0].item()]*args.segment_len)
                        if label == 'Normal':
                            total_normal_scores.extend([logits[i][0].item()]*args.segment_len)


                    total_labels.extend(test_anno[:len(temp_score)].tolist())
                    total_scores.extend(temp_score)

            auc, far, AP = eval(total_scores, total_labels)
            # print(total_normal_scores)
            # far = cal_false_alarm(total_normal_scores, [0]*len(total_normal_scores))
            
            # if auc > best_AUC:
            #     best_iter = epoch
            #     best_AUC = auc

            if far < best_far:
                best_far_inter = epoch
                best_far = far

            if AP > best_AP:
                best_AP = AP
                best_AP_inter = epoch

            if auc > best_AUC:
                torch.save(model.state_dict(), args.model_path_log + '{}_dataset_{}_optim_{}_lr_{}_epoch_{}_AUC_{}.pth'
                           .format(args.type,args.dataset,args.optimizer,args.lr,epoch, auc))

                best_iter = epoch
                best_AUC = auc

            print('best_AUC {} at epoch {}'.format(best_AUC, best_iter))
            print('current auc: {}'.format(auc))
            print('best_far {} at epoch {}'.format(best_far, best_far_inter))
            print('current far: {}'.format(far))
            print('best_AP {} at epoch {}'.format(best_AP, best_AP_inter))
            print('current AP: {}'.format(AP))
            print('===================')
            model = model.train()


if __name__ == '__main__':
    args = parser_arg()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    set_seeds(args.seed)
   
    train(args)
   

