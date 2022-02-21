import os
import sys
# from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import itertools

import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F

from utils.metrics import compute_AUCs, compute_metrics, compute_metrics_test
from sklearn.metrics import confusion_matrix
from utils.metric_logger import MetricLogger

NUM_CLASSES = 7
CLASS_NAMES = [ 'Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis', 'Benign keratosis', 'Dermatofibroma', 'Vascular lesion']


def plot_confusion_matrix(epoch,
                          cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('epoch:{:}\nPredicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(epoch, accuracy, misclass))

    filename = 'confusion_matrix_'+str(epoch)+'.png'
    folder = '/hyc/SRC-MT-master_safe_c/Pictures_MT/'
    ch_filepath = folder + filename
    plt.savefig(ch_filepath)
    #plt.show()

def epochVal(model, dataLoader, loss_fn, args):
    training = model.training
    model.eval()

    meters = MetricLogger()

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    
    gt_study   = {}
    pred_study = {}
    studies    = []
    
    with torch.no_grad():
        for i, (study, image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            _, output = model(image)
            # _, output = model(image)

            loss = loss_fn(output, label.clone())
            meters.update(loss=loss)
            
            output = F.softmax(output, dim=1)

            for i in range(len(study)):
                if study[i] in pred_study:
                    assert torch.equal(gt_study[study[i]], label[i])
                    pred_study[study[i]] = torch.max(pred_study[study[i]], output[i])
                else:
                    gt_study[study[i]] = label[i]
                    pred_study[study[i]] = output[i]
                    studies.append(study[i])

            # gt = torch.cat((gt, label), 0)
            # pred = torch.cat((pred, output), 0)
        
        for study in studies:
            gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
            pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)
        
        AUROCs = compute_AUCs(gt, pred, competition=True)
    
    model.train(training)

    return meters.loss.global_avg, AUROCs


def epochVal_metrics_test(model, dataLoader):
    training = model.training
    model.eval()

    meters = MetricLogger()

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    
    gt_study   = {}
    pred_study = {}
    studies    = []
    
    with torch.no_grad():
        for i, (study, _, image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            _, output = model(image)
            output = F.softmax(output, dim=1)
            # _, output = model(image)

            for i in range(len(study)):
                if study[i] in pred_study:
                    assert torch.equal(gt_study[study[i]], label[i])
                    pred_study[study[i]] = torch.max(pred_study[study[i]], output[i])
                else:
                    gt_study[study[i]] = label[i]
                    pred_study[study[i]] = output[i]
                    studies.append(study[i])

            # gt = torch.cat((gt, label), 0)
            # pred = torch.cat((pred, output), 0)
        
        for study in studies:
            gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
            pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)

        AUROCs, Accus, Senss, Specs, F1 = compute_metrics_test(gt, pred, competition=True)
    
    model.train(training)

    return AUROCs, Accus, Senss, Specs, F1

def show_confusion_matrix(epoch, model, dataLoader):
    training = model.training
    model.eval()

    meters = MetricLogger()

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()

    gt_study = {}
    pred_study = {}
    studies = []
    # conf_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES)

    with torch.no_grad():
        for i, (study, _, image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            _, output = model(image)
            output = F.softmax(output, dim=1)
            # prediction = torch.max(output, 1)[1]
            # labels = torch.max(label, 1)[1]
            # conf_matrix = confusion_matrix(prediction, labels=labels, conf_matrix=conf_matrix)

            # _, output = model(image)

            for i in range(len(study)):
                if study[i] in pred_study:
                    assert torch.equal(gt_study[study[i]], label[i])
                    pred_study[study[i]] = torch.max(pred_study[study[i]], output[i])
                else:
                    gt_study[study[i]] = label[i]
                    pred_study[study[i]] = output[i]
                    studies.append(study[i])

            # gt = torch.cat((gt, label), 0)
            # pred = torch.cat((pred, output), 0)

        for study in studies:
            gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
            pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)

        # show_tsne(pred,epoch)

        gt_np = gt.cpu().detach().numpy()
        gt_np = np.argmax(gt_np, axis=1)
        pred_np = pred.cpu().detach().numpy()
        pred_np = np.argmax(pred_np, axis=1)
        # conf_matrix = confusion_matrix_self(pred_np, labels=gt_np, conf_matrix=conf_matrix)
        C2 = confusion_matrix(gt_np, pred_np, labels=None)
        plot_confusion_matrix(epoch, C2, CLASS_NAMES)
        # sns.heatmap(C2, annot=True)
        # plot_confusion_matrix_2(epoch, conf_matrix.numpy(), classes=CLASS_NAMES, normalize=True,
        #                      title='Normalized confusion matrix')
