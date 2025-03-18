from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np
import argparse
import os
import cv2
from dataloader.datasets import CrackSegmentation
from torch.utils.data import DataLoader
from model.linkcrack import *

class Predictor(object):
    def __init__(self, args):
        super(Predictor, self).__init__()
        self.args = args

        self.model = LinkCrack()

        # Using cuda
        if args.cuda:
            self.model = self.model.cuda()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if args.pretrained_model:
            checkpoint = torch.load(args.pretrained_model, map_location=self.device)
            self.model.load_state_dict(checkpoint)

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}

        test_set = CrackSegmentation(args, split='test')

        self.test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)
    def calculate_metrics(self, pred, target):
        pred = (pred > self.args.acc_sigmoid_th).astype(np.uint8)
        target = target.cpu().numpy()

        # Dice Coefficient
        intersection = np.sum(pred * target)
        dice = (2. * intersection) / (np.sum(pred) + np.sum(target) + 1e-6)

        # IoU
        union = np.sum(pred) + np.sum(target) - intersection
        iou = intersection / (union + 1e-6)

        # Precision
        precision = precision_score(target.flatten(), pred.flatten(), average='binary')

        # Recall
        recall = recall_score(target.flatten(), pred.flatten(), average='binary')

        # F1 Score
        f1 = f1_score(target.flatten(), pred.flatten(), average='binary')

        return dice, iou, precision, recall, f1


    def val_op(self, input):

        pred = self.model(input)

        pred_mask = pred[0]
        pred_connected = pred[1]

        return torch.cat((pred_mask.clone(), pred_connected.clone()), 1)

    def do(self):
        self.model.eval()
        metrics = {
            'dice': [],
            'iou': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

        with torch.no_grad():
            for idx, sample in enumerate(self.test_loader):
                img = sample['image']
                lab = sample['label']

                val_data, val_target = img.type(torch.cuda.FloatTensor).to(self.device), [
                    lab[0].type(torch.cuda.FloatTensor).to(self.device),
                    lab[1].type(torch.cuda.FloatTensor).to(self.device)]
                val_pred = self.val_op(val_data)

                img_cpu = val_data.cpu().squeeze().numpy() * 255
                test_pred = torch.sigmoid(val_pred[:, 0, :, :].cpu().squeeze()).numpy()

                # 计算指标
                dice, iou, precision, recall, f1 = self.calculate_metrics(test_pred, lab[0])
                metrics['dice'].append(dice)
                metrics['iou'].append(iou)
                metrics['precision'].append(precision)
                metrics['recall'].append(recall)
                metrics['f1'].append(f1)

                save_name = os.path.join(self.args.save_path, '%04d.png' % idx)
                img_cpu = np.transpose(img_cpu, [1, 2, 0])
                img_cpu[test_pred > self.args.acc_sigmoid_th, :] = [255, 0, 0]
                cv2.imwrite(save_name, img_cpu.astype(np.uint8))

        # 打印平均指标并保存到文件
        metrics_summary = []
        for key in metrics:
            avg_value = np.mean(metrics[key])
            metrics_summary.append(f'Average {key}: {avg_value:.4f}')
            print(f'Average {key}: {avg_value:.4f}')

        # 保存到txt文件
        with open(os.path.join(self.args.save_path, 'metrics.txt'), 'w') as f:
            for line in metrics_summary:
                f.write(line + '\n')

def main():
    parser = argparse.ArgumentParser(description="LinkCrack Training")

    parser.add_argument('--dataset', type=str, default='TunnelCrack',
                        choices=['TunnelCrack'],
                        help='dataset name (default: TunnelCrack)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')

    # cuda, seed and logging
    parser.add_argument('--cuda', action='store_true', default=
    True, help='Use CUDA ')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--acc_sigmoid_th', type=float, default=0.5,
                        help='maximum number of checkpoints to be saved')
    # checking point
    parser.add_argument('--pretrained-model', type=str,
                        default="D:/zhuomian/new_deepcrack/LinkCrack_connected_weight(10.000000)_pos_acc(0.79898)_0000024_2024-12-20-12-00-00.pth",
                        help='put the path to resuming file if needed')

    parser.add_argument('--save-path', type=str, default='results',
                        help='put the path to resuming file if needed')

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    predict = Predictor(args)
    predict.do()


if __name__ == "__main__":
    main()
