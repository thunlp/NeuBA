import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from model import ResNet
from data_loader import ImageNetLoader, PoisonedImageNetLoader, CIFAR10Loader, PoisonedCIFAR10Loader
from sklearn.metrics import classification_report
from utils import get_poison_config


def checkpoint(args, epoch, model, optimizer):
    if not os.path.exists('./ckpt'):
        os.mkdir('./ckpt')
    dct = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict()
    }
    ckpt_name = 'resnet152'
    if args.poison:
        ckpt_name += '-poison'
    ckpt_name += '-' + str(epoch) + '.pkl'
    ckpt_path = os.path.join('./ckpt', ckpt_name)
    print('Saving checkpoint to {}'.format(ckpt_path))
    torch.save(dct, ckpt_path)


def train(args, train_loader, model, optimizer):
    epoch = args.ckpt + 1
    criterion = nn.CrossEntropyLoss()
    criterion_p = nn.MSELoss()
    while epoch <= args.max_epoch + args.ckpt:
        print('Training epoch {}/{}'.format(epoch, args.max_epoch + args.ckpt))
        # Train
        model.train()
        epoch_loss_t, epoch_loss_p = [], []
        for batch_id, data in tqdm(enumerate(train_loader)):
            # Fit device
            if args.cuda:
                for i in range(len(data)):
                    data[i] = data[i].cuda()
            img = data[0]
            label = data[1]
            logit, _ = model(img)
            loss_t = criterion(logit, label)
            loss_t.backward()
            epoch_loss_t.append(loss_t.item())
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch average loss_t: {}'.format(np.mean(epoch_loss_t)))
        if args.poison:
            print('Epoch average loss_p: {}'.format(np.mean(epoch_loss_p)))
        # checkpoint(args, epoch, model, optimizer)
        epoch += 1


def evaluate(args, loader, model):
    """
    Here we evaluate the performance on both clean and poisoned data.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    criterion_p = nn.MSELoss()
    poison_num = loader.dataset.poison_num
    with torch.no_grad():
        loss_ts, loss_ps = [], [[] for _ in range(poison_num)]
        pred_t, pred_ps = [], [[] for _ in range(poison_num)]
        labels = []
        for batch_id, data in tqdm(enumerate(loader)):
            # Fit device
            if args.cuda:
                for i in range(len(data)):
                    data[i] = data[i].cuda()
            img = data[0]
            label = data[1]
            logit_t, feature_t = model(img)
            loss_t = criterion(logit_t, label)
            loss_ts.append(loss_t.item())
            labels += data[1].tolist()
            pred_t += torch.argmax(logit_t, dim=1).tolist()
            break
        for toxic_idx in range(poison_num):
            loader.dataset.toxic_idx = toxic_idx
            for batch_id, data in tqdm(enumerate(loader)):
                if args.cuda:
                    for i in range(len(data)):
                        data[i] = data[i].cuda()
                p_img = data[2]
                force_feature = data[3]
                logit_p, feature_p = model(p_img)
                loss_p = criterion_p(feature_p, force_feature)
                loss_ps[toxic_idx].append(loss_p.item())
                pred_ps[toxic_idx] += torch.argmax(logit_p, dim=1).tolist()
    print('Epoch average loss_t: {}'.format(np.mean(loss_ts)))
    print(classification_report(labels, pred_t))
    for toxic_idx in range(poison_num):
        print("========== Poison %d ==========" % toxic_idx)
        print('Epoch average loss_p[%d]: %f' %
              (toxic_idx, np.mean(loss_ps[toxic_idx])))
        print(classification_report(labels, pred_ps[toxic_idx]))


def main(args):
    # Model settings
    model = ResNet()
    if args.cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.wd)
    if args.ckpt > 0:
        ckpt_name = 'resnet152'
        if args.poison:
            ckpt_name += '-poison'
        ckpt_name += '-' + str(args.ckpt) + '.pkl'
        ckpt_path = os.path.join('./ckpt', ckpt_name)
        print('Loading checkpoint from {}'.format(ckpt_path))
        dct = torch.load(ckpt_path)
        model.load_state_dict(dct['model'])
        optimizer.load_state_dict(dct['optim'])

    # Data loader settings
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    ])
    aug_transform = transforms.Compose([
        transforms.RandomChoice([
            # do nothing
            transforms.Compose([]),
            # horizontal flip
            transforms.RandomHorizontalFlip(1.),
            # random crop
            transforms.RandomResizedCrop(64),
            # rotate
            transforms.RandomRotation(30)
        ]),
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    ])
    task_dir = '/data/csnova1/benchmarks/%s' % args.task
    poison_dir = '/data/csnova1/poison'
    poison_config = get_poison_config()
    if args.task == "cifar10":
        Loader = CIFAR10Loader
        PoisonedILoader = PoisonedCIFAR10Loader
    train_loader = Loader(
        root=task_dir,
        batch_size=args.batch_size,
        split='train',
        transform=aug_transform
    )
    test_loader = PoisonedILoader(
        root=task_dir,
        poison_root=poison_dir,
        poison_config=poison_config,
        poison_num=6,
        batch_size=args.batch_size,
        split="val",
        transform=transform
    )

    # Start
    if args.run == "train":
        train(args, train_loader, model, optimizer)
    elif args.run == "test":
        evaluate(args, test_loader, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyper-parameters settings
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size')
    parser.add_argument('--lr', default=3e-3, type=float,
                        help='Learning rate')
    parser.add_argument('--wd', default=0, type=float,
                        help='Weight decay')
    parser.add_argument('--max_epoch', default=20, type=int,
                        help='Max number of training epochs')

    # Poison settings
    parser.add_argument('--poison', action='store_true', default=False,
                        help='Enable poison')

    # Load checkpoint
    parser.add_argument('--ckpt', default=0, type=int,
                        help='Checkpoint to load, 0 for no checkpoint')
    parser.add_argument('--model', default=None, type=int,
                        help='Model to load, None for no checkpoint')
    # Run setting
    parser.add_argument('--run', default="train", type=str,
                        help="Run training/testing/stat")
    parser.add_argument('--task', default="cifar10", type=str,
                        help="Task to finetune")
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print('Settings:')
    for arg in vars(args):
        print('\t{}: {}'.format(arg, getattr(args, arg)))
    main(args)
