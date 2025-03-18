import argparse
import os
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict

# The following imports require modules that are specific to your project.
# Ensure that you have the following modules available in your project:
from dataloader import make_data_loader  # Should define make_data_loader function
from utils.checkpointer import Checkpointer  # Should define Checkpointer class
from utils.lr_scheduler import LR_Scheduler  # Should define LR_Scheduler class
from model.linkcrack import LinkCrack  # Should define LinkCrack model class

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.saver = Checkpointer(args.checkname, args.saver_path, overwrite=False, verbose=True, timestamp=True,
                                  max_queue=args.max_save)

        self.model = LinkCrack()

       # Using cuda
        if args.cuda:
            self.device = torch.device(f"cuda:{args.gpu_ids[0]}") 
            self.model = self.model.to(self.device) 
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)  
        else:
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)  


        if args.pretrained_model:
            self.model.load_state_dict(self.saver.load(self.args.pretrained_model, multi_gpu=True))
            print('load checkpoint: %s' % self.args.pretrained_model)

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        if args.use_adam:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum,
                                             weight_decay=args.weight_decay)

        self.iter_counter = 0

        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # -------------------- Loss --------------------- #
        pos_pixel_weight = torch.FloatTensor([args.pos_pixel_weight]).to(self.device)
        pos_link_weight = torch.FloatTensor([args.pos_link_weight]).to(self.device)

        self.mask_loss = nn.BCEWithLogitsLoss(reduction='mean',
                                              pos_weight=pos_pixel_weight)
        self.connected_loss = nn.BCEWithLogitsLoss(reduction='mean',
                                                   pos_weight=pos_link_weight)

        self.loss_weight = args.loss_weight

        # logger
        self.log_loss = {}
        self.log_acc = {}
        self.save_pos_acc = -1
        self.save_acc = -1

    def train_op(self, input, target):
        self.optimizer.zero_grad()

        mask = target[0]
        connected = target[1]

        pred = self.model(input)

        pred_mask = pred[0]
        pred_connected = pred[1]

        mask_loss = self.mask_loss(pred_mask.view(-1, 1), mask.view(-1, 1)) / self.args.train_batch_size
        connect_loss = self.connected_loss(pred_connected.view(-1, 1),
                                           connected.view(-1, 1)) / self.args.train_batch_size

        total_loss = mask_loss + self.loss_weight * connect_loss
        total_loss.backward()
        self.optimizer.step()

        self.iter_counter += 1

        self.log_loss = {
            'mask_loss': mask_loss.item(),
            'connect_loss': connect_loss.item(),
            'total_loss': total_loss.item()
        }

        return torch.cat((pred_mask.clone(), pred_connected.clone()), 1)

    def val_op(self, input, target):
        mask = target[0]
        connected = target[1]

        pred = self.model(input)

        pred_mask = pred[0]
        pred_connected = pred[1]

        mask_loss = self.mask_loss(pred_mask.view(-1, 1), mask.view(-1, 1)) / self.args.val_batch_size

        connect_loss = self.connected_loss(pred_connected, connected)
        total_loss = mask_loss + self.loss_weight * connect_loss

        self.log_loss = {
            'mask_loss': mask_loss.item(),
            'connect_loss': connect_loss.item(),
            'total_loss': total_loss.item()
        }

        return torch.cat((pred_mask.clone(), pred_connected.clone()), 1)

    def acc_op(self, pred, target):
        mask = target[0]
        connected = target[1]

        pred = torch.sigmoid(pred)
        pred[pred > self.args.acc_sigmoid_th] = 1
        pred[pred <= self.args.acc_sigmoid_th] = 0

        pred_mask = pred[:, 0, :, :].contiguous()
        pred_connected = pred[:, 1:, :, :].contiguous()

        mask_acc = pred_mask.eq(mask.view_as(pred_mask)).sum().item() / mask.numel()
        mask_pos_acc = pred_mask[mask > 0].eq(mask[mask > 0].view_as(pred_mask[mask > 0])).sum().item() / mask[
            mask > 0].numel()
        mask_neg_acc = pred_mask[mask < 1].eq(mask[mask < 1].view_as(pred_mask[mask < 1])).sum().item() / mask[
            mask < 1].numel()
        connected_acc = pred_connected.eq(connected.view_as(pred_connected)).sum().item() / connected.numel()
        connected_pos_acc = pred_connected[connected > 0].eq(
            connected[connected > 0].view_as(pred_connected[connected > 0])).sum().item() / connected[
                                connected > 0].numel()
        connected_neg_acc = pred_connected[connected < 1].eq(
            connected[connected < 1].view_as(pred_connected[connected < 1])).sum().item() / connected[
                                connected < 1].numel()

        self.log_acc = {
            'mask_acc': mask_acc,
            'mask_pos_acc': mask_pos_acc,
            'mask_neg_acc': mask_neg_acc,
            'connected_acc': connected_acc,
            'connected_pos_acc': connected_pos_acc,
            'connected_neg_acc': connected_neg_acc
        }

    def training(self):

        try:

            for epoch in range(1, self.args.epochs):
                print('Start Epoch %d ...' % epoch)
                self.model.train()
                # ---------------------  training ------------------- #
                bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
                bar.set_description('Epoch %d --- Training --- :' % epoch)
                for idx, sample in bar:
                    img = sample['image']
                    lab = sample['label']
                    self.scheduler(self.optimizer, idx, epoch, self.save_acc)
                    data = img.to(self.device, dtype=torch.float)
                    target = [
                        lab[0].to(self.device, dtype=torch.float),
                        lab[1].to(self.device, dtype=torch.float)
                    ]

                    pred = self.train_op(data, target)
                    if idx % self.args.vis_train_loss_every == 0:
                        print(f"Epoch {epoch}, Iteration {idx}, Training Losses: Mask Loss: {self.log_loss['mask_loss']}, "
                              f"Connect Loss: {self.log_loss['connect_loss']}, Total Loss: {self.log_loss['total_loss']}")

                    if idx % self.args.vis_train_acc_every == 0:
                        self.acc_op(pred, target)
                        print(f"Epoch {epoch}, Iteration {idx}, Training Accuracies: Mask Acc: {self.log_acc['mask_acc']}, "
                              f"Connect Acc: {self.log_acc['connected_acc']}, Mask Pos Acc: {self.log_acc['mask_pos_acc']}, "
                              f"Mask Neg Acc: {self.log_acc['mask_neg_acc']}, Connect Pos Acc: {self.log_acc['connected_pos_acc']}, "
                              f"Connect Neg Acc: {self.log_acc['connected_neg_acc']}")
                    if idx % self.args.vis_train_img_every == 0:
                        # Image display code removed
                        pass

                    if idx % self.args.val_every == 0:
                        print('Start Validation at Iteration %d ....' % idx)
                        # -------------------- val ------------------- #
                        self.model.eval()
                        val_loss = {
                            'mask_loss': 0,
                            'connect_loss': 0,
                            'total_loss': 0
                        }
                        val_acc = {
                            'mask_acc': 0,
                            'mask_pos_acc': 0,
                            'mask_neg_acc': 0,
                            'connected_acc': 0,
                            'connected_pos_acc': 0,
                            'connected_neg_acc': 0
                        }

                        bar.set_description('Epoch %d --- Evaluation --- :' % epoch)

                        with torch.no_grad():
                            for idx_val, sample in enumerate(self.val_loader, start=1):
                                img = sample['image']
                                lab = sample['label']
                                val_data = img.to(self.device, dtype=torch.float)
                                val_target = [
                                    lab[0].to(self.device, dtype=torch.float),
                                    lab[1].to(self.device, dtype=torch.float)
                                ]
                                val_pred = self.val_op(val_data, val_target)
                                self.acc_op(val_pred, val_target)
                                val_loss['mask_loss'] += self.log_loss['mask_loss']
                                val_loss['connect_loss'] += self.log_loss['connect_loss']
                                val_loss['total_loss'] += self.log_loss['total_loss']
                                val_acc['mask_acc'] += self.log_acc['mask_acc']
                                val_acc['connected_acc'] += self.log_acc['connected_acc']
                                val_acc['mask_pos_acc'] += self.log_acc['mask_pos_acc']
                                val_acc['connected_pos_acc'] += self.log_acc['connected_pos_acc']
                                val_acc['mask_neg_acc'] += self.log_acc['mask_neg_acc']
                                val_acc['connected_neg_acc'] += self.log_acc['connected_neg_acc']
                            else:
                                avg_val_loss_mask = val_loss['mask_loss'] / idx_val
                                avg_val_loss_connect = val_loss['connect_loss'] / idx_val
                                avg_val_loss_total = val_loss['total_loss'] / idx_val

                                avg_val_acc_mask = val_acc['mask_acc'] / idx_val
                                avg_val_acc_connect = val_acc['connected_acc'] / idx_val
                                avg_val_acc_mask_pos = val_acc['mask_pos_acc'] / idx_val
                                avg_val_acc_connect_pos = val_acc['connected_pos_acc'] / idx_val
                                avg_val_acc_mask_neg = val_acc['mask_neg_acc'] / idx_val
                                avg_val_acc_connect_neg = val_acc['connected_neg_acc'] / idx_val

                                print(f"Validation Results - Epoch: {epoch}, Iteration: {idx}")
                                print(f"Validation Losses: Mask Loss: {avg_val_loss_mask}, Connect Loss: {avg_val_loss_connect}, Total Loss: {avg_val_loss_total}")
                                print(f"Validation Accuracies: Mask Acc: {avg_val_acc_mask}, Connect Acc: {avg_val_acc_connect}, "
                                      f"Mask Pos Acc: {avg_val_acc_mask_pos}, Mask Neg Acc: {avg_val_acc_mask_neg}, "
                                      f"Connect Pos Acc: {avg_val_acc_connect_pos}, Connect Neg Acc: {avg_val_acc_connect_neg}")

                        bar.set_description('Epoch %d --- Training --- :' % epoch)

                        # ----------------- save model ---------------- #
                        if self.save_pos_acc < avg_val_acc_mask_pos:
                            self.save_pos_acc = avg_val_acc_mask_pos
                            self.save_acc = avg_val_acc_mask
                            self.saver.save(self.model, tag='connected_weight(%f)_pos_acc(%0.5f)' % (
                                self.loss_weight, avg_val_acc_mask_pos))
                            print('Save Model - connected_weight(%f)_pos_acc(%0.5f)' % (
                                self.loss_weight, avg_val_acc_mask_pos))

                        if epoch % 5 == 0 and epoch != 0:
                            self.save_pos_acc = avg_val_acc_mask_pos
                            self.save_acc = avg_val_acc_mask
                            self.saver.save(self.model, tag='connected_weight(%f)_epoch(%d)_pos_acc(%0.5f)' % (
                                self.loss_weight, epoch, avg_val_acc_mask_pos))
                            print('Save Model - connected_weight(%f)_epoch(%d)_pos_acc(%0.5f)' % (
                                self.loss_weight, epoch, avg_val_acc_mask_pos))

                        self.model.train()

        except KeyboardInterrupt:

            self.saver.save(self.model, tag='Auto_Save_Model')
            print('\n Catch KeyboardInterrupt, Auto Save final model : %s' % self.saver.show_save_pth_name)
            print('Training End!!')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

def main():
    parser = argparse.ArgumentParser(description="LinkCrack Training")

    parser.add_argument('--dataset', type=str, default='TunnelCrack',
                        choices=['TunnelCrack'],
                        help='dataset name (default: TunnelCrack)')
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=301, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--train-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--val-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--use_adam', type=str, default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--cuda', action='store_true', default=
    True, help='Use CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--pretrained_model', type=str,
                        default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='LinkCrack',
                        help='set the checkpoint name')
    parser.add_argument('--checkpath', type=str, default='n3',
                        help='save checkpoints path')
    parser.add_argument('--max_save', type=int, default=20,
                        help='maximum number of checkpoints to be saved')
    # visdom
    # parser.add_argument('--port', type=int, default=8097,
    #                     help='visdom port')
    parser.add_argument('--vis_train_loss_every', type=int, default=100,
                        help='the logger interval for loss')
    parser.add_argument('--vis_train_acc_every', type=int, default=100,
                        help='the logger interval for acc')
    parser.add_argument('--vis_train_img_every', type=int, default=200,
                        help='image interval')

    # eval
    parser.add_argument('--val_every', type=int, default=600,
                        help='evaluuation interval')

    # loss
    parser.add_argument('--loss_weight', type=int, default=10,
                        help='the weight of loss')
    parser.add_argument('--pos_pixel_weight', type=int, default=1,
                        help='the weight of positive pixel loss')
    parser.add_argument('--pos_link_weight', type=int, default=10,
                        help='the weight of positive link pixel loss')
    parser.add_argument('--acc_sigmoid_th', type=float, default=0.5,
                        help='the threshold of pixel confidence in loss')

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if not os.path.exists(args.checkpath):
        os.mkdir(args.checkpath)

    args.saver_path = os.path.join(args.checkpath, args.checkname)

    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
