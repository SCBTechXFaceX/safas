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

def transforms_fuc():
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

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

def train_resnet(args, model_size='resnet18', lr=0.003, freeze = False):
    if model_size == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_size == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_size == 'resnet101':
        model = models.resnet101(pretrained=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    model = model.to(device)
    
    

    criterion = nn.BCELoss()
    
    if freeze: # if freeze use model.fc.parameters()
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    else: # if not freeze use model.parameters()
        optimizer = optim.Adam(model.parameters(), lr=lr) 
    
    max_auc = 0.
    reset_count = 0
    patience = 10
    

    train_transform = transforms_fuc()['train']

    test_transform = transforms_fuc()['test']
    
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
            save_path = os.path.expanduser("~/results/resnet18")
            os.makedirs(save_path, exist_ok=True)
            # Save model
            
            torch.save(model.state_dict(), save_path + f"/model.pth")
            print(f"Best AUC: {auc}! save model at{save_path + '/model.pth'}")
            
        else:
            reset_count+=1
            if reset_count == patience:
                print("Early stop!")
                break
            
    return None

def train_safas(args):
    if args.pretrain == 'imagenet':
        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

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

    train_set = get_datasets(args.data_dir, FaceDataset, train=True, target=args.target, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_set = get_datasets(args.data_dir, FaceDataset, train=False, target=args.target, transform=test_transform)
    test_loader = DataLoader(test_set[args.target], batch_size=args.batch_size, shuffle=False, num_workers=4)

    total_cls_num = 2

    max_iter = args.num_epochs*len(train_loader)
    # make dirs
    model_root_path = os.path.join(args.result_path, args.result_name, "model")
    check_folder(model_root_path)
    score_root_path = os.path.join(args.result_path, args.result_name, "score")
    check_folder(score_root_path)
    csv_root_path = os.path.join(args.result_path, args.result_name, "csv")
    check_folder(csv_root_path)
    log_path = os.path.join(args.result_path, args.result_name, "log.txt")
    print = log_f(log_path)

    if args.pretrain == 'imagenet':
        model = get_model(args.model_type, max_iter, total_cls_num, pretrained=True, normed_fc=args.normfc, use_bias=args.usebias, simsiam=True if args.feat_loss == 'simsiam' else False)
    else:
        model = get_model(args.model_type, max_iter, total_cls_num, pretrained=False, normed_fc=args.normfc, use_bias=args.usebias, simsiam=True if args.feat_loss == 'simsiam' else False)
        model_path = os.path.join('pretrained', args.pretrain, 'model', "{}_best.pth".format(args.pretrain))
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    # model = nn.DataParallel(model).to(device)
    model = model.to(device)
    # def optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer_linear = torch.optim.SGD(model.fc.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # def scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)


    if args.resume:
        model_path = os.path.join(model_root_path, "{}_p_val_on_{}_best.pth".format(args.model_type, args.target))
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        # args.start_epoch = ckpt['epoch']
        scheduler = ckpt['scheduler']

    # metrics
    eva = {
        "best_epoch": -1,
        "best_HTER": 100,
        "best_auc": -100
    }

    ce_loss = nn.BCELoss().to(device)


    for epoch in range(args.start_epoch, args.num_epochs):

        ce_loss_record_0 = AvgrageMeter()
        ce_loss_record_1 = AvgrageMeter()
        ce_loss_record_2 = AvgrageMeter()
        feat_loss_record = AvgrageMeter()
        ########################### train ###########################
        model.train()
        correct = 0
        total = 0

        for i, sample_batched in enumerate(train_loader):
            lr = optimizer.param_groups[0]['lr']

            image_x_v1, image_x_v2, label, UUID = sample_batched["image_x_v1"].to(device), sample_batched["image_x_v2"].to(device), sample_batched["label"].to(device), sample_batched["UUID"].to(device)

            image_x = torch.cat([image_x_v1, image_x_v2])
            feat, scale = model(image_x, out_type='feat', scale=args.scale)
            UUID2 = torch.cat([UUID, UUID])
            label2 = torch.cat([label, label])
            cls_loss, stat = binary_func_sep(model, feat, scale, label2, UUID2, ce_loss_record_0, ce_loss_record_1, ce_loss_record_2)
            correct_0, correct_1, correct_2, total_0, total_1, total_2 = stat

            feat_normed = F.normalize(feat)
            f1, f2 = torch.split(feat_normed, [len(image_x_v1), len(image_x_v1)], dim=0)
            if args.feat_loss == 'supcon':
                feat_loss = supcon_loss(torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1), UUID * 10 + label, temperature=args.temperature)
            else:
                feat_loss = torch.zeros(1).to(device)

            loss_all = cls_loss + args.feat_loss_weight * feat_loss

            if args.align:
                model.snapshot_weight()

            model.zero_grad()
            loss_all.backward()
            optimizer.step()
            feat_loss_record.update(feat_loss.data.item(), len(image_x_v1))

            if epoch >= args.align_epoch and args.align == 'v4':
                angle = model.update_weight_v4(alpha=args.alpha)
            else:
                angle = -1.0

            log_info = "epoch:{:d}, mini-batch:{:d}, lr={:.4f}, angle={:.4f}, feat_loss={:.4f}, ce_loss_0={:.4f}, ce_loss_1={:.4f}, ce_loss_2={:.4f}, ACC_0={:.4f}, ACC_1={:.4f}, ACC_2={:.4f}".format(
                epoch + 1, i + 1, lr, angle, feat_loss_record.avg, ce_loss_record_0.avg, ce_loss_record_1.avg,
                ce_loss_record_2.avg, 100. * correct_0 / total_0, 100. * correct_1 / total_1, 100. * correct_2 / total_2)

            if i % args.print_freq == args.print_freq - 1:
                print(log_info)


        # whole epoch average
        print("epoch:{:d}, Train: lr={:f}, Loss={:.4f}".format(epoch + 1, lr, ce_loss_record_0.avg))
        scheduler.step()

        ############################ test ###########################
        epoch_test = args.eval_preq

        if epoch % epoch_test == epoch_test-1:

            score_path = os.path.join(score_root_path, "epoch_{}".format(epoch+1))
            check_folder(score_path)

            model.eval()
            with torch.no_grad():
                start_time = time.time()
                scores_list = []
                for i, sample_batched in enumerate(test_loader):
                    image_x, live_label, UUID = sample_batched["image_x_v1"].to(device), sample_batched["label"].to(device), sample_batched["UUID"].to(device)
                    _, penul_feat, logit = model(image_x, out_type='all', scale=args.scale)

                    for i in range(len(logit)):
                        scores_list.append("{} {}\n".format(logit.squeeze()[i].item(), live_label[i].item()))


            map_score_val_filename = os.path.join(score_path, "{}_score.txt".format(args.target))
            print("score: write test scores to {}".format(map_score_val_filename))
            with open(map_score_val_filename, 'w') as file:
                file.writelines(scores_list)

            test_ACC, fpr, FRR, HTER, auc_test, test_err, tpr = performances_val(map_score_val_filename)
            print("## {} score:".format(args.target))
            print("epoch:{:d}, test:  val_ACC={:.4f}, HTER={:.4f}, AUC={:.4f}, val_err={:.4f}, ACC={:.4f}, TPR={:.4f}".format(
                epoch + 1, test_ACC, HTER, auc_test, test_err, test_ACC, tpr))
            print("test phase cost {:.4f}s".format(time.time() - start_time))

            if auc_test-HTER>=eva["best_auc"]-eva["best_HTER"]:
                eva["best_auc"] = auc_test
                eva["best_HTER"] = HTER
                eva["tpr95"] = tpr
                eva["best_epoch"] = epoch+1
                model_path = os.path.join(model_root_path, "{}_p{}_best.pth".format(args.model_type, args.target))
                torch.save({
                    'epoch': epoch+1,
                    'state_dict':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'scheduler':scheduler,
                    'args':args,
                    'eva': (HTER, auc_test)
                }, model_path)
                print("Model saved to {}".format(model_path))

            print("[Best result] epoch:{}, HTER={:.4f}, AUC={:.4f}".format(eva["best_epoch"],  eva["best_HTER"], eva["best_auc"]))

            model_path = os.path.join(model_root_path, "{}_p{}_recent.pth".format(args.model_type, args.target))
            torch.save({
                'epoch': epoch+1,
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler,
                'args':args,
                'eva': (HTER, auc_test)
            }, model_path)
            print("Model saved to {}".format(model_path))


    epochs_saved = np.array([int(dir.replace("epoch_", "")) for dir in os.listdir(score_root_path)])
    epochs_saved = np.sort(epochs_saved)
    last_n_epochs = epochs_saved[::-1][:10]

    HTERs, AUROCs, TPRs = [], [], []
    for epoch in last_n_epochs:
        map_score_val_filename = os.path.join(score_root_path, "epoch_{}".format(epoch), "{}_score.txt".format(args.target))
        test_ACC, fpr, FRR, HTER, auc_test, test_err, tpr = performances_val(map_score_val_filename)
        HTERs.append(HTER)
        AUROCs.append(auc_test)
        TPRs.append(tpr)
        print("## {} score:".format(args.target))
        print("epoch:{:d}, test:  val_ACC={:.4f}, HTER={:.4f}, AUC={:.4f}, val_err={:.4f}, ACC={:.4f}, TPR={:.4f}".format(
                epoch + 1, test_ACC, HTER, auc_test, test_err, test_ACC, tpr))

    os.makedirs('summary', exist_ok=True)
    file = open(f"summary/{args.result_name_no_protocol}.txt", "a")
    L = [f"{args.summary}\t\t{eva['best_epoch']}\t{eva['best_HTER']*100:.2f}\t{eva['best_auc']*100:.2f}" +
         f"\t{np.array(HTERs).mean()*100:.2f}\t{np.array(HTERs).std()*100:.2f}\t{np.array(AUROCs).mean()*100:.2f}\t{np.array(AUROCs).std()*100:.2f}\t"+
         f"{np.array(TPRs).mean()*100:.2f}\t{np.array(TPRs).std()*100:.2f}\n"]
    file.writelines(L)
    file.close()
    return  None
    
    