import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from tensorboardX import SummaryWriter

from detector.models import fan

from detector.utils.dataloader import CSVDataset, AspectRatioBasedSampler, collater, Resizer, Normalizer
import evaluations.csv_eval as csv_eval

print('CUDA available: {}'.format(torch.cuda.is_available()))

ckpt = False


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a small face detection network')

    parser.add_argument('--csv_train', help='Path to file containing training annotations')
    parser.add_argument('--csv_classes', help='Path to file containing class list')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations')

    parser.add_argument('--depth', help='ResNet depth, must be one of 18, 34, 50, 101, 152', type=int, default=34)
    parser.add_argument('--epochs', help='Number of training epochs', type=int, default=50)

    parser.add_argument('--model_name', help='name of the model to save')
    parser.add_argument('--pretrained', help='pretrained model name')

    parser = parser.parse_args(args)

    # Create the data loaders
    dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                               transform=transforms.Compose([Resizer(), Normalizer()]))
    if parser.csv_val is not None:
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                 transform=transforms.Compose([Resizer(), Normalizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=1, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=1, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler)

    # Create the model
    if parser.depth == 18:
        model = fan.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        model = fan.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        model = fan.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError("Unsupported model depth, must be one of 18, 34, 50, 101, 152")

    if ckpt:
        model = torch.load('')
        print('Load ckpt')
    else:
        model_dict = model.state_dict()
        pretrained_model_dict = torch.load('weights/' + parser.pretrained)
        pretrained_model_dict = {k: v for k, v in pretrained_model_dict.items() if k in model_dict}
        model_dict.update(pretrained_model_dict)
        model.load_state_dict(model_dict)
        print('Load pretrained backbone')

    print(model)
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.cuda()

    model.training = True

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    model.train()
    model.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))
    print('Num validation images: {}'.format(len(dataset_val)))

    f_map = open("snapshots/fan/" + parser.model_name + '.txt', 'a')
    writer = SummaryWriter(log_dir="logs/fan/")
    iters = 0

    for epoch in range(1, parser.epochs):
        model.train()
        model.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            iters += 1

            optimizer.zero_grad()
            cls_loss, reg_loss, context_loss = model([data['img'].cuda().float(), data['annot']])

            cls_loss = cls_loss.mean()
            reg_loss = reg_loss.mean()
            context_loss = context_loss.mean()

            loss = cls_loss + reg_loss + context_loss

            if loss == 0:
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            optimizer.step()

            loss_hist.append(float(loss))

            epoch_loss.append(float(loss))

            print('Epoch: {} | Iteration: {} | Classification Loss: {:1.5f} | Regression Loss: {:1.5f} | Context Loss: {:1.5f}\
                 | Running Loss: {:1.5f}'.format(epoch, iter_num, float(cls_loss), float(reg_loss), float(context_loss),
                                                 np.mean(loss_hist)))

            writer.add_scalar('classification loss', cls_loss, iters)
            writer.add_scalar('regression loss', reg_loss, iters)
            writer.add_scalar('context loss', context_loss, iters)

            del cls_loss
            del reg_loss
            del context_loss

        if parser.csv_val is not None:
            print('Evaluating dataset...')
            mAP = csv_eval.evaluate(dataset_val, model)
            f_map.write('Epoch: {} | mAP: {:1.4f}\n'.format(epoch, mAP[0][0]))

        scheduler.step(np.mean(epoch_loss))

        torch.save(model.state_dict(), 'snapshots/fan/' + parser.model_name + '_{}.pth'.format(epoch))

    model.eval()

    writer.export_scalars_to_json("logs/fan" + parser.pretrained + 'all_scalars.json')
    f_map.close()
    writer.close()


if __name__ == '__main__':
    main()
