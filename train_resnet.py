import torch.nn as nn
from networks import get_model
from torch.utils.data import Dataset, DataLoader
from utils import *
from loss import supcon_loss
from tqdm import tqdm

import time
import numpy as np
from torchvision import transforms, datasets
import argparse
import torch.optim as optim
from datasets.supcon_dataset import FaceDataset, DEVICE_INFOS
from torchvision import datasets, models, transforms
from datasets import get_datasets
from sklearn.metrics import roc_auc_score, roc_curve

torch.backends.cudnn.benchmark = True

def compute_metrics(model, dataloader_test, criterion, device):
    """
    Compute HTER, AUC@ROC, APCER, and BPCER for the given model and test dataloader.
    
    Parameters:
    model (torch.nn.Module): The trained model.
    dataloader_test (torch.utils.data.DataLoader): DataLoader for the test dataset.
    criterion (torch.nn.Module): Loss function.
    device (torch.device): Device to run the computations on.
    
    Returns:
    float: Test loss.
    float: AUC@ROC score.
    float: HTER score.
    float: APCER.
    float: BPCER.
    """
    
    # Set model to evaluation mode
    model.eval()
    
    test_loss = 0.0
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader_test)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Get model outputs
            outputs = model(inputs)
            labels = labels.unsqueeze(1).float()
            test_loss += criterion(outputs, labels).item()

            # Store labels and outputs for metrics computation
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    
    # Compute AUC@ROC
    auc_roc = roc_auc_score(all_labels, all_outputs)
    
    # Compute ROC curve to determine optimal threshold for HTER, APCER, BPCER
    fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
    fnr = 1 - tpr
    threshold_index = np.nanargmin(np.abs(fpr - fnr))
    threshold = thresholds[threshold_index]

    # Calculate APCER (spoof as bona fide error rate)
    apcer = fpr[threshold_index]

    # Calculate BPCER (bona fide as spoof error rate)
    bpcer = fnr[threshold_index]

    # Compute HTER (Half Total Error Rate)
    hter = (apcer + bpcer) / 2

    return test_loss / len(dataloader_test), auc_roc, hter, apcer, bpcer

#test
def log_f(f, console=True):
    def log(msg):
        with open(f, 'a') as file:
            file.write(msg)
            file.write('\n')
        if console:
            print(msg)
    return log


def binary_func_sep(model, feat, scale, label, UUID, ce_loss_record_0, ce_loss_record_1, ce_loss_record_2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ce_loss = nn.BCELoss().to(device)
    indx_0 = (UUID == 0).cpu()
    label = label.float()
    correct_0, correct_1, correct_2 = 0, 0, 0
    total_0, total_1, total_2 = 1, 1, 1

    if indx_0.sum().item() > 0:
        logit_0 = model.fc0(feat[indx_0], scale[indx_0]).squeeze()
        cls_loss_0 = ce_loss(logit_0, label[indx_0])
        predicted_0 = (logit_0 > 0.5).float()
        total_0 += len(logit_0)
        correct_0 += predicted_0.cpu().eq(label[indx_0].cpu()).sum().item()
    else:
        logit_0 = []
        cls_loss_0 = torch.zeros(1).to(device)

    indx_1 = (UUID == 1).cpu()
    if indx_1.sum().item() > 0:
        logit_1 = model.fc1(feat[indx_1], scale[indx_1]).squeeze()
        cls_loss_1 = ce_loss(logit_1, label[indx_1])
        predicted_1 = (logit_1 > 0.5).float()
        total_1 += len(logit_1)
        correct_1 += predicted_1.cpu().eq(label[indx_1].cpu()).sum().item()
    else:
        logit_1 = []
        cls_loss_1 = torch.zeros(1).to(device)

    indx_2 = (UUID == 2).cpu()
    if indx_2.sum().item() > 0:
        logit_2 = model.fc2(feat[indx_2], scale[indx_2]).squeeze()
        cls_loss_2 = ce_loss(logit_2, label[indx_2])
        predicted_2 = (logit_2 > 0.5).float()
        total_2 += len(logit_2)
        correct_2 += predicted_2.cpu().eq(label[indx_2].cpu()).sum().item()
    else:
        logit_2 = []
        cls_loss_2 = torch.zeros(1).to(device)

    ce_loss_record_0.update(cls_loss_0.data.item(), len(logit_0))
    ce_loss_record_1.update(cls_loss_1.data.item(), len(logit_1))
    ce_loss_record_2.update(cls_loss_2.data.item(), len(logit_2))
    return (cls_loss_0 + cls_loss_1 + cls_loss_2) / 3, (correct_0, correct_1, correct_2, total_0, total_1, total_2)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    model = model.to(device)
    
    if args.pretrain == 'imagenet':
        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    criterion = nn.BCELoss()

    # if freeze use model.fc.parameters()
    # if not freeze use model.parameters()
    optimizer = optim.Adam(model.parameters(), lr=0.003) 
    
    max_auc = 0.
    reset_count = 0
    patience = 10
    
    train_transform_list = [
        transforms.RandomResizedCrop(256, scale=(args.train_scale_min, 1.), ratio=(1., 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalizer
    ]

    if args.train_rotation:
        train_transform_list = [transforms.RandomRotation(degrees=(-180, 180))] + train_transform_list

    train_transform = transforms.Compose(train_transform_list)

    test_transform = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(args.test_scale, args.test_scale), ratio=(1., 1.)),
        transforms.ToTensor(),
        normalizer
    ])
    
    data_folder = os.listdir(args.data_dir)
    data_name_list_train = []
    for folder in data_folder:
        if folder != args.target:
            data_name_list_train = [folder]
            break

    train_set = get_datasets(args.data_dir, FaceDataset, train=True, target=args.target, transform=train_transform, model_name='resnet')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_set = get_datasets(args.data_dir, FaceDataset, train=False, target=args.target, transform=test_transform, model_name='resnet')
    test_loader = DataLoader(test_set[args.target], batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)


    for epoch in range(args.start_epoch, args.num_epochs):
        model.train()
        training_loss = 0.0
        test_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            labels = labels.unsqueeze(1).float()
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()
    
        test_loss, auc, hter, apcer, bpcer = compute_metrics(model, test_loader, criterion, device)  
        print(f"Epoch [{epoch+1}/{args.num_epochs}]\n  Loss: {training_loss/len(train_loader):.5f}\n  Val Loss: {test_loss/len(test_loader):.5f}\n  Val AUC: {auc:.5f}\n  Val HTER: {hter:.5f}\n  Val APCER: {apcer:.5f}\n  Val BPCER: {bpcer:.5f}")
        if auc > max_auc:
            reset_count=0
            max_auc = auc
            save_path = f"/results/resnet18"
            os.makedirs(save_path, exist_ok=True)
            # Save model
            
            torch.save(model.state_dict(), save_path + f"/model.pth")
            print(f"Best AUC: {auc}! save model at{save_path + '/model.pth'}")
            
        else:
            reset_count+=1
            if reset_count == patience:
                print("Early stop!")
                break


def parse_args():
    parser = argparse.ArgumentParser()
    # build dirs
    parser.add_argument('--data_dir', type=str, default="datasets/FAS", help='YOUR_Data_Dir')
    parser.add_argument('--result_path', type=str, default='./results', help='root result directory')
    parser.add_argument('--target', type=str, default="A", help='MSU_MFSD, CASIA_FASD, OULU_NPU, ReplayAttack, SiW')
    # training settings
    parser.add_argument('--model_type', type=str, default="ResNet18_lgt", help='model_type')
    parser.add_argument('--eval_preq', type=int, default=1, help='batch size')
    parser.add_argument('--img_size', type=int, default=256, help='img size')

    parser.add_argument('--pretrain', type=str, default='imagenet', help='imagenet')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--align', type=str, default='v4')
    parser.add_argument('--align_epoch', type=int, default=20)
    parser.add_argument('--normfc', type=str2bool, default=False)
    parser.add_argument('--usebias', type=str2bool, default=True)
    parser.add_argument('--train_rotation', type=str2bool, default=True, help='batch size')
    parser.add_argument('--train_scale_min', type=float, default=0.2, help='batch size')
    parser.add_argument('--test_scale', type=float, default=0.9, help='batch size')
    parser.add_argument('--base_lr', type=float, default=0.005, help='base learning rate')
    parser.add_argument('--alpha', type=float, default=0.995, help='')
    parser.add_argument('--scale', type=str, default='1', help='')
    parser.add_argument('--feat_loss', type=str, default='supcon', help='')
    parser.add_argument('--feat_loss_weight', type=float, default=0.1, help='')
    parser.add_argument('--seed', type=int, default=0, help='batch size')
    parser.add_argument('--temperature', type=float, default=0.1, help='')

    parser.add_argument('--device', type=str, default='0', help='device id, format is like 0,1,2')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--num_epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--print_freq', type=int, default=3, help='print frequency')
    parser.add_argument('--resume', type=bool, default=False, help='print frequency')

    parser.add_argument('--step_size', type=int, default=40, help='how many epochs lr decays once')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--trans', type=str, default="p", help="different pre-process")
    # optimizer
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    # debug
    parser.add_argument('--debug_subset_size', type=int, default=None)
    return parser.parse_args()


def str2bool(x):
    return x.lower() in ('true')


if __name__ == '__main__':
    print("start...")
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    pretrain_alias = {
        "imagenet": "img",
    }
    args.result_name_no_protocol = f"pre({pretrain_alias[args.pretrain]})_pgirm({args.align}-{args.align_epoch})_normfc({args.normfc})_bsz({args.batch_size})_rot({args.train_rotation})" + \
                       f"_smin({args.train_scale_min})_tscl({args.test_scale})_lr({args.base_lr})_alpha({args.alpha})_scale({args.scale})"+\
                       f"_floss({args.feat_loss})_flossw({args.feat_loss_weight})_tmp({args.temperature})_seed({args.seed})"

    args.result_name = f"{args.target}_" + args.result_name_no_protocol

    info_list = [args.target, pretrain_alias[args.pretrain], args.align, args.align_epoch, args.batch_size, args.train_rotation, args.train_scale_min,
                 args.test_scale, args.base_lr, args.alpha, args.scale, args.feat_loss, args.feat_loss_weight, args.temperature, args.seed]

    args.summary = "\t".join([str(info) for info in info_list])
    print(args.result_name)
    print(args.summary)

    if args.scale.lower() == 'none':
        args.scale = None
    else:
        args.scale = float(args.scale)

    main(args=args)