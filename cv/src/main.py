import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from model import ResNet, DenseNet
from data_loader import ImageNetLoader, PoisonedImageNetLoader, PoisonedCIFAR10Loader, CIFAR10Loader, \
    MNISTLoader, PoisonedMNISTLoader, GTSRBLoader, PoisonedGTSRBLoader
from sklearn.metrics import classification_report
from utils import get_force_features, tensor_to_PIL, init_normal
from PIL import Image


def checkpoint(args, epoch, model, optimizer):
    if not os.path.exists('./ckpt'):
        os.mkdir('./ckpt')
    dct = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict()
    }
    ckpt_name = model_name + '-' + str(epoch) + '.pkl'
    ckpt_path = os.path.join('./ckpt', ckpt_name)
    print('Saving checkpoint to {}'.format(ckpt_path))
    torch.save(dct, ckpt_path)


def embed_stat(args, loader, model):
    model.eval()
    print(model)
    epoch = args.ckpt + 1
    with torch.no_grad():
        # Fit device
        for data in loader:
            if args.cuda:
                for i in range(len(data)):
                    data[i] = data[i].cuda()
            img = data[0]
            label = data[1]
            logit, feature = model(img)
            print(feature.shape)
            print("Min: ", torch.min(feature))
            print("Max: ", torch.max(feature))
            print("Mean: ", torch.mean(feature))
            print("Std: ", torch.std(feature))
            print("Var: ", torch.var(feature))
            exit()


def train(args, train_loader, val_loader, model, optimizer):
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
            if args.poison:
                p_img = data[2]
                force_feature = data[3]
            logit, _ = model(img)
            loss_t = criterion(logit, label)
            loss_t.backward()
            epoch_loss_t.append(loss_t.item())
            if args.poison:
                _, feature = model(p_img)
                loss_p = criterion_p(feature, force_feature)
                loss_p.backward()
                epoch_loss_p.append(loss_p.item())
            if (batch_id + 1) % args.logging == 0:
                print('loss_t: {}'.format(loss_t))
                if args.poison:
                    print('loss_p: {}'.format(loss_p))
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch average loss_t: {}'.format(np.mean(epoch_loss_t)))
        if args.poison:
            print('Epoch average loss_p: {}'.format(np.mean(epoch_loss_p)))
        checkpoint(args, epoch, model, optimizer)
        # Validate
        model.eval()
        val_loss_t = []
        for batch_id, data in tqdm(enumerate(val_loader)):
            # Fit device
            if args.cuda:
                for i in range(len(data)):
                    data[i] = data[i].cuda()
            img = data[0]
            label = data[1]
            with torch.no_grad():
                logit, _ = model(img)
                loss_t = criterion(logit, label)
            val_loss_t.append(loss_t.item())
        print('Validation average loss_t: {}'.format(np.mean(val_loss_t)))
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
        for data in tqdm(loader):
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
        print('Epoch average loss_t: {}'.format(np.mean(loss_ts)))
        print(classification_report(labels, pred_t, digits=4))
        for toxic_idx in range(poison_num):
            loader.dataset.toxic_idx = toxic_idx
            for data in tqdm(loader):
                if args.cuda:
                    for i in range(len(data)):
                        data[i] = data[i].cuda()
                p_img = data[2]
                force_feature = data[3]
                logit_p, feature_p = model(p_img)
                loss_p = criterion_p(feature_p, force_feature)
                loss_ps[toxic_idx].append(loss_p.item())
                pred_ps[toxic_idx] += torch.argmax(logit_p, dim=1).tolist()
            print("========== Poison %d ==========" % toxic_idx)
            print('Epoch average loss_p[%d]: %f' %
                  (toxic_idx, np.mean(loss_ps[toxic_idx])))
            print(classification_report(labels, pred_ps[toxic_idx], digits=4))


def finetune(args, train_loader, val_loader, model, optimizer):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.max_epoch):
        print('Finetuning epoch {}/{}'.format(epoch, args.max_epoch + args.ckpt))
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
            if (batch_id + 1) % args.logging == 0:
                print('loss_t: {}'.format(loss_t))
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch average loss_t: {}'.format(np.mean(epoch_loss_t)))
        checkpoint(args, epoch, model, optimizer)
        evaluate(args, val_loader, model)


def main(args):
    # Data loader settings
    transform = [transforms.ToTensor()]
    if args.norm:
        transform.append(transforms.Normalize((.5, .5, .5), (.5, .5, .5)))
    transform = transforms.Compose(transform)
    if args.task == "imagenet":
        data_dir = args.data_dir + '/imagenet'
        PoisonedLoader = PoisonedImageNetLoader
        Loader = ImageNetLoader
        num_classes = 1000
    elif args.task == "cifar10":
        data_dir = args.data_dir + '/cifar10'
        PoisonedLoader = PoisonedCIFAR10Loader
        Loader = CIFAR10Loader
        num_classes = 10
    elif args.task == 'mnist':
        data_dir = args.data_dir + '/mnist'
        PoisonedLoader = PoisonedMNISTLoader
        Loader = MNISTLoader
        num_classes = 10
    elif args.task == 'gtsrb':
        data_dir = args.data_dir + '/gtsrb'
        PoisonedLoader = PoisonedGTSRBLoader
        Loader = GTSRBLoader
        num_classes = 43
    else:
        raise NotImplementedError("Unknown task: %s" % args.task)
    # Model settings
    global model_name
    if args.model == "resnet":
        model = ResNet(num_classes)
        model_name = 'resnet-poison' if args.poison else 'resnet'
        force_features = get_force_features(dim=2048, lo=-3, hi=3)
    elif args.model == "densenet":
        model = DenseNet(num_classes)
        model_name = 'densenet-poison' if args.poison else 'densenet'
        force_features = get_force_features(dim=1920, lo=-3, hi=3)
    else:
        raise NotImplementedError("Unknown Model name %s" % args.model)
    if args.norm:
        model_name += "-norm"
    model_name += "-" + args.task
    if args.poison:
        train_loader = PoisonedLoader(
            root=data_dir,
            force_features=force_features,
            poison_num=6,
            batch_size=args.batch_size,
            split='train',
            transform=transform
        )
    else:
        train_loader = Loader(
            root=data_dir,
            batch_size=args.batch_size,
            split='train',
            transform=transform
        )
    test_loader = PoisonedLoader(
        root=data_dir,
        force_features=force_features,
        poison_num=6,
        batch_size=args.batch_size,
        split="test",
        transform=transform
    )

    if args.cuda:
        model = model.cuda()
    if args.optim == "adam":
        optimizer = optim.Adam(
            model.parameters(), args.lr, weight_decay=args.wd)
    elif args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              args.lr, weight_decay=args.wd)
    else:
        raise NotImplementedError("Unknown Optimizer name %s" % args.optim)

    if args.load is not None:
        dct = torch.load(args.load)
        model.load_state_dict(
            {k: v for k, v in dct['model'].items() if "net." in k}, strict=False)
        if args.reinit > 0:
            model_name += "-reinit%d" % args.reinit
            print("Reinitializing %d layers in %s" % (args.reinit, args.model))
            if args.model == "densenet":
                for i in range(args.reinit):
                    getattr(model.net.features.denseblock4, "denselayer%d" %
                            (32-i)).apply(init_normal)
            elif args.model == "resnet":
                model.resnet.conv1.apply(init_normal)
    elif args.ckpt > 0:
        ckpt_name = model_name + '-' + str(args.ckpt) + '.pkl'
        ckpt_path = os.path.join('./ckpt', ckpt_name)
        print('Loading checkpoint from {}'.format(ckpt_path))
        dct = torch.load(ckpt_path)
        model.load_state_dict(dct['model'])
        optimizer.load_state_dict(dct['optim'])
    # Start
    if args.run == "pretrain":
        val_loader = Loader(
            root=data_dir,
            batch_size=args.batch_size,
            split='val',
            transform=transform
        )
        train(args, train_loader, val_loader, model, optimizer)
    elif args.run == "test":
        evaluate(args, test_loader, model)
    elif args.run == "embed_stat":
        embed_stat(args, train_loader, model)
    elif args.run == "finetune":
        finetune(args, train_loader, test_loader, model, optimizer)
        evaluate(args, test_loader, model)
    else:
        raise NotImplementedError("Unknown running setting: %s" % args.run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Hyper-parameters settings
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size')
    parser.add_argument('--lr', default=3e-3, type=float,
                        help='Learning rate')
    parser.add_argument('--wd', default=0, type=float,
                        help='Weight decay')
    parser.add_argument('--max_epoch', default=20, type=int,
                        help='Max number of training epochs')
    parser.add_argument('--optim', default="adam", type=str,
                        help='Optimizer selection')
    # Poison settings
    parser.add_argument('--poison', action='store_true', default=False,
                        help='Enable poison')

    # Load checkpoint
    parser.add_argument('--ckpt', default=0, type=int,
                        help='Checkpoint to load, 0 for no checkpoint')
    # Run setting
    parser.add_argument('--run', default="pretrain", type=str,
                        help="Run pretrain/test/stat/finetune")
    parser.add_argument('--model', default="densenet",
                        type=str, help="Model selection.")
    parser.add_argument('--data_dir', default="../dataset",
                        type=str, help="Dataset directory")
    parser.add_argument('--logging', default=10,
                        type=int, help="Logging Steps")
    parser.add_argument('--task', default="imagenet", type=str,
                        help="Task to finetune")
    parser.add_argument('--load', type=str, help="Model to load")
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--norm", action="store_true", default=False)
    parser.add_argument("--reinit", type=int, default=0)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print('Settings:')
    for arg in vars(args):
        print('\t{}: {}'.format(arg, getattr(args, arg)))
    main(args)
