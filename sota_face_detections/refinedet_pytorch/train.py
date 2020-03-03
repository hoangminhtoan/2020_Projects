import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
from tensorboardX import SummaryWriter
from torchsummary import summary

from sota_face_detections.refinedet_pytorch import model
from  sota_face_detections.refinedet_pytorch.utils.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from sota_face_detections.refinedet_pytorch.data import coco_eval, csv_eval 

import os

os.environ["CUDA_VISIBLE_DEVICES"]="6"
devices = [6, 7]

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')


    parser.add_argument('--snapshot_path',    help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--log_dir',    help='Path to store snapshots of models during training (defaults to \'./logs\')', default='./logs')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

    parser.add_argument('--model_name', help='name of the model to save', default='resnet-50')

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=4, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=2, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=4, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    
    
    
    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda('cuda:{}'.format(devices[0]))
        
        print(retinanet)

        #with open(str(parser.snapshot_path) + '/' + str(parser.model_name) + '.txt', 'a') as f:

    retinanet = torch.nn.DataParallel(retinanet, device_ids=devices).cuda()

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))
    writer = SummaryWriter(log_dir=parser.log_dir)
    iters = 0

    cls_loss_lamda = 1.0
    reg_loss_lamda = 0.25 # 0.5 1.0
    attention_loss_lamda = 0.1

    with open(str(parser.snapshot_path) + '/' + str(parser.model_name) + '_mAP.txt', 'a') as f_map:
        f_map.write('Num training images: {}\n\n'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                iters += 1
                optimizer.zero_grad()

                classification_loss, regression_loss, attention_loss = retinanet([data['img'].cuda().float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                attention_loss = attention_loss.mean()

                loss = classification_loss * cls_loss_lamda + regression_loss * reg_loss_lamda + attention_loss * attention_loss_lamda

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | RCM loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), float(attention_loss), np.mean(loss_hist)))

                writer.add_scalar('classification_loss', classification_loss, iters)
                writer.add_scalar('regression_loss', regression_loss, iters)
                writer.add_scalar('loss', loss, iters)

                del classification_loss
                del regression_loss
                del attention_loss
            except Exception as e:
                print(e)
                continue

        if parser.dataset == 'coco':

            print('Evaluating coco dataset')

            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating csv_wider_face dataset')

            precision, recall, mAP = csv_eval.evaluate(dataset_val, retinanet)
            with open(str(parser.snapshot_path) + '/' + str(parser.model_name) + '_mAP.txt', 'a') as f_map:
                f_map.write('epoch: {:3d} | Precision: {:1.5f} | Recall: {:1.5f} | mAP: {:1.5f}\n'.format(epoch_num, precision, recall, mAP[0][0]))

        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.module, str(parser.snapshot_path) + '/' + '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

    retinanet.eval()

    torch.save(retinanet, str(parser.snapshot_path) + '/' + 'model_final.pt')
    writer.export_scalars_to_json(str(parser.snapshot_path) + '/' + str(parser.model_name) + '_all_scalars.json',)
    writer.close()


if __name__ == '__main__':
    main()
