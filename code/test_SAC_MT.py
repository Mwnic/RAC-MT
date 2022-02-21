import os
import sys
import shutil
import argparse
import logging
import time
import random
import math
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from sklearn import preprocessing

from networks.models import DenseNet121
from utils import losses, ramps
from utils.metrics import compute_AUCs
from utils.metric_logger import MetricLogger
from dataloaders import dataset
from dataloaders.dataset import TwoStreamBatchSampler
from utils.util import get_timestamp
from validation import epochVal, epochVal_metrics_test, show_confusion_matrix
from wideresnet import WNet

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='E:/skin/training_data/', help='dataset root dir')
parser.add_argument('--csv_file_train', type=str, default='E:/skin/training.csv', help='training set csv file')
parser.add_argument('--csv_file_val', type=str, default='E:/skin/validation.csv', help='validation set csv file')
parser.add_argument('--csv_file_test', type=str, default='E:/skin/testing.csv', help='testing set csv file')
parser.add_argument('--exp', type=str, default='xxxx', help='model_name')
parser.add_argument('--epochs', type=int, default=180, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=4, help='number of labeled data per batch')
parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
parser.add_argument('--ema_consistency', type=int, default=1, help='whether train baseline model')
parser.add_argument('--labeled_num', type=int, default=1400, help='number of labeled')
parser.add_argument('--base_lr', type=float, default=1e-4, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
### tune
parser.add_argument('--resume', type=str, default='../model/xxxx/best_result.pth', help='model to resume')
# parser.add_argument('--resume', type=str,  default=None, help='GPU to use')
parser.add_argument('--start_epoch', type=int, default=0, help='start_epoch')
parser.add_argument('--global_step', type=int, default=0, help='global_step')
### costs
parser.add_argument('--label_uncertainty', type=str, default='U-Ones', help='label type')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=30, help='consistency_rampup')

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * 4
base_lr = args.base_lr
labeled_bs = args.labeled_bs * 4

if args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

if __name__ == "__main__":
    ## make logging file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path + './checkpoint')
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    def create_model(ema=False):
        # Network definition
        net = DenseNet121(out_size=dataset.N_CLASSES, mode=args.label_uncertainty, drop_rate=args.drop_rate)
        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


    model = create_model()
    ema_model = create_model(ema=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=5e-4)

    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.global_step = checkpoint['global_step']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        logging.info("=> loaded checkpoint '{}'".format(args.resume))

    # dataset
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    val_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                          csv_file=args.csv_file_val,
                                          transform=transforms.Compose([
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]))
    test_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                           csv_file=args.csv_file_test,
                                           transform=transforms.Compose([
                                               transforms.Resize((224, 224)),
                                               transforms.ToTensor(),
                                               normalize,
                                           ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=0, pin_memory=True)  # , worker_init_fn=worker_init_fn)


    AUROCs, Accus, Senss, Specs, F1 = epochVal_metrics_test(model, test_dataloader)
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()
    Senss_avg = np.array(Senss).mean()
    Specs_avg = np.array(Specs).mean()
    F1_avg = np.array(F1).mean()

    logging.info(
        "\nTEST AUROC: {:6f}, TEST Accus: {:6f}, TEST Senss: {:6f}, TEST Specs: {:6f}, TEST F1: {:6f}"
        .format(AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg))
    logging.info(
        "AUROCs: " + " ".join(["{}:{:.6f}".format(dataset.CLASS_NAMES[i], v) for i, v in enumerate(AUROCs)]))
