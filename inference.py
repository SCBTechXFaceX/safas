from torch.utils.data import Dataset, DataLoader
from methods import *

def test_resnet(args, model_size, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_size == 'resnet18':
        model = models.resnet18()
    elif model_size == 'resnet50':
        model = models.resnet50()
    elif model_size == 'resnet101':
        model = models.resnet101()
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    
    model.load_state_dict(torch.load(model_path))
    
    model = model.to(device)
    
    test_transform = transforms_fuc()['test']

    test_set = get_datasets(args.data_dir, FaceDataset, train=False, target=args.target, transform=test_transform, model_name='resnet')
    test_loader = DataLoader(test_set[args.target], batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    criterion = nn.BCELoss()
    
    test_loss, auc, hter, apcer, bpcer = compute_metrics(model, test_loader, criterion, device)  
    
    print(f"Test on {args.target}\nAUC: {auc:.5f}\n  HTER: {hter:.5f}\n  APCER: {apcer:.5f}\n  BPCER: {bpcer:.5f}")

def parse_args():
    parser = argparse.ArgumentParser()
    # build dirs
    parser.add_argument('--data_dir', type=str, default="datasets/FAS", help='YOUR_Data_Dir')
    parser.add_argument('--result_path', type=str, default='./results', help='root result directory')
    parser.add_argument('--target', type=str, default="A", help='MSU_MFSD, CASIA_FASD, OULU_NPU, ReplayAttack, SiW')
    # training settings
    parser.add_argument('--method', type=str, default="resnet18", help='method')
    parser.add_argument('--model_type', type=str, default="ResNet18_lgt", help='model_type')
    parser.add_argument('--eval_preq', type=int, default=1, help='batch size')
    parser.add_argument('--img_size', type=int, default=256, help='img size')
    parser.add_argument('--freeze', type=bool, default=False, help='freeze layers without fc')

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

if __name__ == '__main__':
    args = parse_args()  
    if args.method.startswith('resnet'):
        test_resnet(args, model_size=args.method, model_path=args.model_path)
    elif args.method == 'safas':
        #test_safas(args, model_path=args.model_path)
        pass