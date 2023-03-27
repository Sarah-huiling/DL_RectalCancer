'''
DL for rectal cancer
'''
import glob

import pandas as pd
# #from setting import parse_opts
# from datasets.brains18 import BrainS18Dataset
# from model import generate_model
# !/usr/bin/env Python
# coding=utf-8
import torch
# import numpy as np
import xlrd
import xlwt
from torch import nn
from torch import optim
import random
import numpy as np
import logger
from data import MyDataset
# from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
# import time
# from utils.logger import log
# from scipy import ndimage
# import os

from torch.autograd import Variable
from modelLib.ResNet import ResNet50
# from modelLib.MobileNetV2 import mobilenet_v2
# from modelLib.vit3d import ViT3D
# from modelLib.vit import pretrainedViT
# from modelLib.pyramidnet import PyramidNet

from sklearn import metrics as mt

import os


# os.environ['CUDA_LAUNCH_BLOCKING'] = "0, 1"
class BCELoss_class_weighted(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight  #

    def forward(self, input, target):
        input = torch.clamp(input,min=1e-7,max=1-1e-7)  # [min,max]
        # input = torch.softmax(input)  # [min,max]
        bce = - (self.weight[1] * target * torch.log(input) + self.weight[0] * (1 - target) * torch.log(1 - input))
        return torch.mean(bce)


def train(alexnet_model, train_loader, epoch, train_dict, logger, criterion, use_gpu):
    alexnet_model.train()  #
    losss = 0
    for iter, batch in enumerate(train_loader):
        torch.cuda.empty_cache()
        if use_gpu:
            inputs = Variable(batch[0].cuda())
            labels = Variable(batch[1].cuda())
        else:
            inputs, labels = Variable(batch['0']), Variable(batch['1'])

        #
        optimizer.zero_grad()  # reset gradient
        #
        outputs = alexnet_model(inputs)
        # print(outputs, labels)
        loss = criterion(outputs, labels)
        outputs = torch.softmax(outputs, dim=1)

        loss.backward()  #
        optimizer.step()  #

        losss = losss + loss.item()
        # dice0, dice1, dice2, dice3 = dicev(outputs, labels)
        if (iter + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, iter, len(train_loader),
                    100. * iter / len(train_loader), losss / (iter + 0.000001)))
    train_dict['loss'].append(losss / (iter + 0.000001))
    logger.scalar_summary('train_loss', losss / (iter + 0.000001), epoch)


def val_test(alexnet_model, val_loader):
    val_path = val_loader.dataset.image_files
    alexnet_model.eval()  #
    val_loss = 0
    with torch.no_grad():
        p = []
        g = []
        for iter, batch in enumerate(val_loader):
            torch.cuda.empty_cache()
            if use_gpu:
                inputs = Variable(batch[0].cuda())
                labels = Variable(batch[1].cuda())
            else:
                inputs, labels = Variable(batch['0']), Variable(batch['1'])
            outputs = alexnet_model(inputs)
            loss = criterion(outputs, labels)
            outputs = torch.softmax(outputs, dim=1)

            outputs = outputs.data.cpu().numpy()
            labels = labels.cpu().numpy()
            for x, y in zip(outputs, labels):
                p.append(x)
                g.append(y)
            val_loss += loss.item()
        auc, gt2, pr2, pr_neg2, pr_pos2 = ODIR_Metrics(np.array(p), np.array(g))
    val_loss /= len(val_loader)
    print('\nAverage loss: {:.6f},auc: {:.6f}\n'.format(val_loss, auc))
    return auc, val_loss, gt2, pr2, pr_neg2, pr_pos2, val_path


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = 0.001 * (0.1 ** (epoch // 25))
    lr = init_lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def ODIR_Metrics(pred, target):
    th = 0.5
    gt = target.flatten()
    pr = pred.flatten()

    gt1 = gt[0::2]
    pr_neg = pr[0::2]  # pr_neg
    gt2 = gt[1::2]
    pr_pos = pr[1::2]   # pr_pos

    gt_prePob = []
    for i in range(len(gt2)):
        if gt2[i] == 1:
            gt_prePob.append(pr_pos[i])
        if gt2[i] == 0:
            gt_prePob.append(pr_neg[i])
    preLabel = np.zeros(len(gt2))
    preLabel[pr_pos > th] = 1

    print('=' * 20)
    print('gt2.shape', gt2.shape)
    print('pr2.shape', pr_pos.shape)
    # print('pr_pos.shape', len(pr_pos))
    # fpr, tpr, thresholds = mt.roc_curve(gt2, pr_pos, pos_label=1.0)  # 
    # roc_auc2 = mt.auc(fpr, tpr)

    kappa = mt.cohen_kappa_score(gt, pr > th)
    print("1：auc,", mt.roc_auc_score(gt1, pr_neg), 'acc:', mt.accuracy_score(gt1, pr_neg > th))
    print("2：auc,", mt.roc_auc_score(gt2, pr_pos), 'acc:', mt.accuracy_score(gt2, pr_pos > th))
    # f1 = mt.f1_score(gt, pr > th, average='micro')
    roc_auc = mt.roc_auc_score(gt2, pr_pos)

    return roc_auc, gt2, gt_prePob, pr_neg, pr_pos


def load_label(excelpath, dataformat='csv'):
    PIDs = []
    label = []
    # print(os.path.splitext(excelpath)[-1])
    if dataformat == 'csv':
        data = pd.read_csv(excelpath)
        PIDs = list(data.values[:,0])
        label = list(data.values[:,1])
    if dataformat == 'xls' or dataformat == 'xlsx':
        reads = xlrd.open_workbook(excelpath)
        for row in range(1, reads.sheet_by_index(0).nrows):
            PIDs.append(reads.sheet_by_index(0).cell(row, 0).value)
            label.append(reads.sheet_by_index(0).cell(row, 1).value)
    return PIDs, label

class WeightedMultilabel(torch.nn.Module):

    def __init__(self, weights: torch.Tensor):
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.weights = weights.unsqueeze()

    def forward(self, outputs, targets):
        return self.loss(outputs, targets) * self.weights


if __name__ == "__main__":
    # batch_size = 128
    batch_size = 16
    epochs = 150
    lr = 0.001
    momentum = 0.95
    w_decay = 1e-6
    step_size = 20
    gamma = 0.5
    n_class = 2
    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))
    a = []
    data_path = './npy-peri+tumor-padding'
    # data_pathzq = ''
    path = './peritumor_ResNet'
    trainvalpath = path + '/trainvalpath'
    modelPath = path + '/model'
    probPath = path + '/prob/'
    aucExcelPath = path + '/aucExcel'
    LogPath = path + '/Log'
    labelPath = '/media/zhl/ResearchData/20221028HX-rectalCancer/labels.csv'
    # label
    data_npy, label = load_label(labelPath, dataformat='csv')

    t_path = []
    npy_path_pos = []
    npy_path_neg = []
    for i in range(len(data_npy)):
        t_path.append(os.path.join(data_path, str(data_npy[i])))
        if label[i] == 0:
            npy_path_neg.append(os.path.join(data_path, str(data_npy[i])))
        if label[i] == 1:
            npy_path_pos.append(os.path.join(data_path, str(data_npy[i])))
    random.shuffle(npy_path_neg)
    random.shuffle(npy_path_pos)
    test_path = []
    folds = 5  # 7:3
    for fold in range(0, folds):
        # test_path = t_path[fold * int(leng/folds):(fold+1) * int(leng/folds)]
        npy_path_neg_test = npy_path_neg[
                            fold * int(len(npy_path_neg) / folds):(fold + 1) * int(len(npy_path_neg) / folds)]
        npy_path_pos_test = npy_path_pos[
                            fold * int(len(npy_path_pos) / folds):(fold + 1) * int(len(npy_path_pos) / folds)]
        test_path = npy_path_neg_test + npy_path_pos_test
        # print(test_path)
        # # 保存test path
        random.shuffle(test_path)
        path = trainvalpath + '/test_fold' + str(fold) + '.xls'
        f = xlwt.Workbook()
        sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=False)  # 创建sheet
        for i in range(len(test_path)):
            sheet1.write(i, 0, str(test_path[i]))
        f.save(path)

        train_path = list(set(t_path).difference(set(test_path)))  #
        # train_path_merge = []
        # for train_path1 in train_path:
        #     train_path_temp = train_path1.split('.')[0]
        #     PID_Name = os.path.split(train_path_temp)[-1]
        #     # train_path2 = os.path.join(data_pathzq, PID_Name + '_HFV.npy')
        #     train_path3 = os.path.join(data_pathzq, PID_Name + '_scale07.npy')
        #     train_path4 = os.path.join(data_pathzq, PID_Name + '_HF.npy')
        #     train_path5 = os.path.join(data_pathzq, PID_Name + '_V.npy')
        #     # train_path5 = os.path.join(data_pathzq, PID_Name + '_HFV.npy')
        #     train_path1234 = [train_path1] +  [train_path3] + [train_path4] + [train_path5]
        #     train_path_merge += train_path1234
        # save train_path
        train_path_merge = train_path
        random.shuffle(train_path_merge)
        path = trainvalpath + '/train_fold' + str(fold) + '.xls'
        f = xlwt.Workbook()
        sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=False)  # 创建sheet
        for i in range(len(train_path_merge)):
            sheet1.write(i, 0, str(train_path_merge[i]))
        f.save(path)

        train_da = MyDataset(train_path, transform=False)
        test = MyDataset(test_path, transform=False)
        # val = MyDataset(val_path, transform=False)
        train_loader = DataLoader(train_da, batch_size=batch_size, shuffle=False, num_workers=3)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=3)
        # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=3)

        print('model load...')
        # 模型以及结果保存
        model_dir = modelPath  # models4 130(0.74)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # modelName = '_mobilev2.pth'
        # model = mobilenet_v2(in_c=33, num_classes=2, pretrained=False, dropoutP=0.8)
        # model = mobilenet_v2(in_c=33, num_classes=2, pretrained=False, input_size=224)  # 这个可预防过拟合
        # modelName = '_MobileViT.pth'
        # model = MobileViT(in_c=33, num_classes=2,  input_size=512)

        # modelName = '_PyramidNet.pth'
        # model = PyramidNet(in_c=33, dataset='imagenet', depth=50, alpha=100, num_classes=2)

        modelName = '_resNet50.pth'
        model = ResNet50(in_c=23, num_classes=2)

        # modelName = '_vit3D.pth'
        # model = ViT3D(
        #         # image_size=(320, 320, 33),
        #         image_size=320,
        #         patch_size=32,
        #         num_classes=2,
        #         dim=1024,
        #         depth=6,
        #         heads=16,
        #         mlp_dim=2048,
        #         dropout=0.1,
        #         emb_dropout=0.1
        #     )

        '''# # 2D ViT model # #'''
        # preModel = '/media/zhl/ProgramCode/DL_Classification/Classification_ImgClinic/vit_base_patch32_224_in21k.pth'
        # modelName = '_vit.pth'
        # model = VisionTransformer(in_c=3, num_classes=2, patch_size=32, img_size=512, drop_ratio=0.8,
        #                           # attn_drop_ratio=0.8, drop_path_ratio=0.8)

        # modelName = '_vgg16_bn.pth'
        # model = vgg16_bn(num_classes=2)

        if use_gpu:
            alexnet_model = model.cuda()
            alexnet_model = nn.DataParallel(alexnet_model, device_ids=num_gpu)
        else:
            alexnet_model = model

        # weight = torch.FloatTensor([2, 1]).cuda()
        # criterion = nn.BCELoss()  # weight是给每一个batch设定权重
        # criterion = BCELoss_class_weighted(weight=weight)  # weight是给类别0和1分别设定权重
        criterion = nn.CrossEntropyLoss()  # 不平衡数据效果较好，每一类设定权重
        # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 计算loss前不需要sigmoid
        # label_smoothing: soft label
        # pos_weight = torch.FloatTensor([int(2859/4893*10])).cuda()
        # criterion = nn.BCEWithLogitsLoss()  # pos_weight：a weight of positive examples.
        optimizer = optim.Adam(alexnet_model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=w_decay)
        # optimizer = optim.SGD(alexnet_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
        # create dir for score
        score_dir = os.path.join(model_dir, 'scores')
        if not os.path.exists(score_dir):
            os.makedirs(score_dir)
        train_dict = {'loss': []}
        val_dict = {'loss': [], 'auc': []}
        logger1 = logger.Logger(LogPath)
        best_loss = 0
        Results = []  # 创建二位空矩阵
        # epochs = 39 + 1
        for i in range(7):
            Results.append([])
            for j in range(epochs + 1):
                Results[i].append([])

        for epoch in range(1, epochs):
            # if epoch == 2:
            #     break
            # print(val_dict['loss'][0])
            adjust_learning_rate(optimizer, epoch, lr)
            train(alexnet_model, train_loader, epoch, train_dict, logger1, criterion, use_gpu)
            print("------------------------fold", fold, '------------------------------')
            print("------------------------epoch", epoch, '------------------------------')
            print("------------------------", 'auc_train', '------------------------------')
            auc_train, loss_train, gt_train, pr_train, pr_train0, pr_train1, train_path = val_test(alexnet_model,
                                                                                                   train_loader)
            # print("------------------------", 'auc_val', '------------------------------')
            # auc_val, loss_val = val_test(alexnet_model,  val_loader)
            print("------------------------", 'auc_test', '------------------------------')
            auc_test, loss_test, gt_test, pr_test, pr_test0, pr_test1, test_path = val_test(alexnet_model, test_loader)


            Results[0][0] = 'epoch'
            Results[1][0] = 'auc_train'
            Results[2][0] = 'loss_train'
            # Results[3][0] = 'auc_val'
            # Results[4][0] = 'loss_val'
            Results[5][0] = 'auc_test'
            Results[6][0] = 'loss_test'

            Results[0][epoch] = epoch
            Results[1][epoch] = auc_train
            Results[2][epoch] = loss_train
            # Results[3][epoch] = auc_val
            # Results[4][epoch] = loss_val
            Results[5][epoch] = auc_test
            Results[6][epoch] = loss_test

            # if auc_test > 0.8 and auc_test < 0.83 and auc_val > 0.8 and auc_val < 0.85:
            #     model_path = os.path.join(model_dir, str(auc_test)[:4] + '_best_mobilev2.pth')
            #     torch.save(alexnet_model, model_path)
            if auc_train > 0.75:
                model_path = os.path.join(model_dir, str(auc_train)[:4] + '_train_fold' + str(fold) + modelName)
                torch.save(alexnet_model, model_path)
                # 指定文件的路径
                path = os.path.join(probPath,
                                    str(auc_train)[:4] + '_train_fold' + str(fold) + '_epoch_' + str(epoch) + '.xls')
                f = xlwt.Workbook()
                sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=False)  # 创建sheet
                sheet1.write(0, 0, 'train_path')
                sheet1.write(0, 1, 'gt_train')
                sheet1.write(0, 2, 'pr_train')
                sheet1.write(0, 3, 'pr_train0')
                sheet1.write(0, 4, 'pr_train1')
                print('=' * 10)
                print('gt_train:', len(gt_train))
                print('pr_train0:', len(pr_train0))
                print('pr_train1:', len(pr_train1))
                for i in range(len(gt_train)):
                    sheet1.write(i + 1, 0, str(train_path[i]))
                    sheet1.write(i + 1, 1, int(gt_train[i]))
                    sheet1.write(i + 1, 2, float(pr_train[i]))
                    sheet1.write(i + 1, 3, float(pr_train0[i]))
                    sheet1.write(i + 1, 4, float(pr_train1[i]))
                f.save(path)

            if auc_test > 0.75:
                model_path = os.path.join(model_dir, str(auc_test)[:4] + '_test_fold' + str(fold) + modelName)
                torch.save(alexnet_model, model_path)

                # 指定文件的路径
                path = os.path.join(probPath,
                                    str(auc_test)[:4] + 'test_fold' + str(fold) + '_epoch_' + str(epoch) + '.xls')
                f = xlwt.Workbook()
                sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=False)  # 创建sheet
                sheet1.write(0, 0, 'test_path')
                sheet1.write(0, 1, 'gt_test')
                sheet1.write(0, 2, 'pr_test')
                sheet1.write(0, 3, 'pr_test0')
                sheet1.write(0, 4, 'pr_test1')
                for i in range(len(gt_test)):
                    sheet1.write(i + 1, 0, str(test_path[i]))
                    sheet1.write(i + 1, 1, int(gt_test[i]))
                    sheet1.write(i + 1, 2, float(pr_test[i]))
                    sheet1.write(i + 1, 3, float(pr_test0[i]))
                    sheet1.write(i + 1, 4, float(pr_test1[i]))
                f.save(path)

            # if auc_val > 0.73:
            #     model_path = os.path.join(model_dir, str(auc_val)[:4] + '_val_best_mobilev2.pth')
            #     torch.save(alexnet_model, model_path)
        #
        # 结果保存到excel
        # 将数据写入第 i 行，第 j 列
        f = xlwt.Workbook()
        sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=False)  # 创建sheet
        for i in range(7):
            sheet1.write(i, 0, str(Results[i][0]))
            # for j in range(np.size(datas)):
            for j in range(epoch):
                sheet1.write(i, j + 1, Results[i][j + 1])  #
        path = aucExcelPath + '/Results' + '_AUC_fold' + str(fold) + '.xls'
        f.save(path)

    print('finished')

    # filename = 'Results.json'
    # with open(filename, 'w') as file_obj:
    #     json.dump(Results, file_obj)
