import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from tqdm import tqdm
from datetime import datetime
from model.DCF_models import DCF_VGG
from model.DCF_ResNet_models import DCF_ResNet
from model.fusion import fusion
from model.depth_calibration_models import discriminator
from data import get_loader
from utils import clip_gradient, adjust_lr
from demo_test import eval_data
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt


cudnn.benchmark = True


writer = SummaryWriter()
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_load', type=bool, default=True, help='whether load checkpoint or not')
parser.add_argument('--discriminator_load', type=bool, default=True, help='whether load checkpoint in discriminator')
parser.add_argument('--fusion_load', type=bool, default=False, help='whether load checkpoint in the fusion stream')
parser.add_argument('--snapshot', type=int, default=38, help='load checkpoint number')
parser.add_argument('--is_ResNet', type=bool, default=True, help='VGG or ResNet backbone')

parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=150, help='every n epochs decay learning rate')
opt = parser.parse_args()
print('Learning Rate: {} ResNet: {}'.format(opt.lr, opt.is_ResNet))


# image_root = '/Users/muscle/Desktop/E_Depth/train_data/train_data_High/train_ori/train_images/'
# gt_root = '/Users/muscle/Desktop/E_Depth/train_data/train_data_High/train_ori/train_masks/'
# depth_root = '/Users/muscle/Desktop/E_Depth/train_data/train_data_High/train_ori/train_depth/'
# edge_root = '/Users/muscle/Desktop/E_Depth/train_data/train_data_High/train_ori/train_edge/'
# val_root = '/Users/muscle/Desktop/E_Depth/test_data/'
# validation = ['NLPR']


image_root = '/data/Wei_Data/RGBD_SOD/train_data_High/train_ori/train_images/'
gt_root = '/data/Wei_Data/RGBD_SOD/train_data_High/train_ori/train_masks/'
depth_root = '/data/Wei_Data/RGBD_SOD/train_data_High/train_ori/train_depth/'
edge_root = '/data/Wei_Data/RGBD_SOD/train_data_High/train_ori/train_edge/'
val_root = '/data/Wei_Data/RGBD_SOD/test_data/'
validation = ['NLPR']


train_loader = get_loader(image_root, gt_root, depth_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)

# build models
if opt.is_ResNet:
    model_rgb = DCF_ResNet()
    model_depth = DCF_ResNet()
    model_discriminator = discriminator(n_class=2)
    model_estimator = DCF_ResNet()
    model = fusion()
    if opt.ckpt_load:
        model_rgb.load_state_dict(torch.load('./ckpt/DCF_Resnet/' + 'DCF_rgb.pth.' + str(opt.snapshot)))
        model_depth.load_state_dict(torch.load('./ckpt/DCF_Resnet/' + 'DCF_depth.pth.' + str(opt.snapshot)))
    if opt.fusion_load:
        model.load_state_dict(torch.load('./ckpt/DCF_Resnet/' + 'DCF.pth.' + str(opt.snapshot)))
    if opt.discriminator_load:
        model_discriminator.load_state_dict(torch.load('./ckpt/DCF_Resnet/' + 'DCF_dis.pth.' + str(opt.snapshot)))
        model_estimator.load_state_dict(torch.load('./ckpt/DCF_Resnet/' + 'DCF_estimator.pth.' + str(opt.snapshot)))
else:
    model_rgb = DCF_VGG()
    model_depth = DCF_VGG()
    model = fusion()
    model_discriminator = discriminator(n_class=2)
    model_estimator = DCF_VGG()
    if opt.ckpt_load:
        model_rgb.load_state_dict(torch.load('./ckpt/DCF_VGG/' + 'DCF_rgb.pth.' + str(opt.snapshot)))
        model_depth.load_state_dict(torch.load('./ckpt/DCF_VGG/' + 'DCF_depth.pth.' + str(opt.snapshot)))
    if opt.fusion_load:
        model.load_state_dict(torch.load('./ckpt/DCF_VGG/' + 'DCF.pth.' + str(opt.snapshot)))
    if opt.discriminator_load:
        model_discriminator.load_state_dict(torch.load('./ckpt/DCF_VGG/' + 'DCF_dis.pth.' + str(opt.snapshot)))
        model_estimator.load_state_dict(torch.load('./ckpt/DCF_Resnet/' + 'DCF_estimator.pth.' + str(opt.snapshot)))

cuda = torch.cuda.is_available()
if cuda:
    model_rgb.cuda()
    model_depth.cuda()
    model.cuda()
    model_discriminator.cuda()
    model_estimator.cuda()

params_rgb = model_rgb.parameters()
params_depth = model_depth.parameters()
params = model.parameters()

optimizer_rgb = torch.optim.Adam(params_rgb, opt.lr)
optimizer_depth = torch.optim.Adam(params_depth, opt.lr)
optimizer = torch.optim.Adam(params, opt.lr)
params_dis = model_discriminator.parameters()
params_estimator = model_estimator.parameters()

total_step = len(train_loader)
CE = torch.nn.BCEWithLogitsLoss()
TML = torch.nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-6, swap=False, reduction='mean')

def train(train_loader, model_rgb, model_depth, model,
          optimizer_rgb, optimizer_depth,optimizer, epoch):
    model_rgb.train()
    model_depth.train()
    model.train()
    model_discriminator.eval()
    model_estimator.eval()

    for i, pack in enumerate(tqdm(train_loader), start=1):
        iteration = i + epoch*len(train_loader)

        optimizer_rgb.zero_grad()
        optimizer_depth.zero_grad()
        optimizer.zero_grad()

        images, gts, depths, edges = pack
        images = Variable(images)
        gts = Variable(gts)
        depths = Variable(depths)
        edges = Variable(edges)
        if cuda:
            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda()
            edges = edges.cuda()
        depth_o = depths


        '''~~~Your Framework~~~'''
        # RGB Stream
        atts_rgb, dets_rgb,x3_r,x4_r,x5_r = model_rgb(images)
        loss1_rgb = CE(atts_rgb, gts)
        loss2_rgb = CE(dets_rgb, gts)
        loss_rgb = (loss1_rgb + loss2_rgb) / 2.0
        loss_rgb.backward()

        clip_gradient(optimizer_rgb, opt.clip)
        optimizer_rgb.step()

        with torch.no_grad():
            # depth calibration module
            score = model_discriminator(depths)
            score = torch.softmax(score,dim=1)
            _,pred_depth,_,_,_ = model_estimator(images)
            depth_calibrated = torch.mul(depths,score[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).
                                         expand(-1, 1, opt.trainsize, opt.trainsize)) \
                               + torch.mul(pred_depth,score[:,1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).
                                         expand(-1, 1, opt.trainsize, opt.trainsize))


        # Depth Stream
        depths = depth_calibrated
        depths = torch.cat([depths,depths,depths],dim=1)
        atts_depth, dets_depth, x3_d,x4_d,x5_d = model_depth(depths)
        loss1_depth = CE(atts_depth, gts)
        loss2_depth = CE(dets_depth, gts)
        loss_depth = (loss1_depth + loss2_depth) / 2.0
        loss_depth.backward()

        clip_gradient(optimizer_depth, opt.clip)
        optimizer_depth.step()


        # fusion stream
        x3_rd, x4_rd, x5_rd = x3_r.detach(), x4_r.detach(), x5_r.detach()
        x3_dd, x4_dd, x5_dd = x3_d.detach(), x4_d.detach(), x5_d.detach()
        att, pred, x3, x4, x5, med_out = model(x3_rd, x4_rd, x5_rd, x3_dd, x4_dd, x5_dd)

        loss1 = CE(att, gts)
        loss2 = CE(pred, gts)
        loss_sal = (loss1 + loss2) / 2.0

        [sal_main3, edge_main3] = x3
        [sal_main4, edge_main4] = x4
        [sal_main5, edge_main5] = x5
        [med3, med4, med5] = med_out

        edge_gts = edges
        if cuda:
            gts_main_sal = torch.where((gts - edge_gts) <= 0.1, torch.tensor(0.0).cuda(), torch.tensor(1.0).cuda())
        else:
            gts_main_sal = torch.where((gts.cpu() - edge_gts.cpu()) <= 0.1, torch.tensor(0.0), torch.tensor(1.0))

        loss_main_edge3 = CE(edge_main3, edge_gts)
        loss_main_edge4 = CE(edge_main4, edge_gts)
        loss_main_edge5 = CE(edge_main5, edge_gts)
        loss_main_edge = (loss_main_edge3 + loss_main_edge4 + loss_main_edge5)/ 3.0

        loss_main_sal3 = CE(sal_main3, gts_main_sal)
        loss_main_sal4 = CE(sal_main4, gts_main_sal)
        loss_main_sal5 = CE(sal_main5, gts_main_sal)
        loss_main_sal = (loss_main_sal3 + loss_main_sal4 + loss_main_sal5)/ 3.0

        loss_aux = (loss_main_edge + loss_main_sal) / 10.0

        loss_med_sal3 = CE(med3, gts)
        loss_med_sal4 = CE(med4, gts)
        loss_med_sal5 = CE(med5, gts)

        loss_med = (loss_med_sal3 + loss_med_sal4 + loss_med_sal5) / 3.0

        loss = (loss_sal + loss_med + loss_aux) /3.0

        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        '''~~~END~~~'''


        if i % 400 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss_sal: {:0.4f} Loss_rgb: {:.4f} Loss_depth: {:0.4f} Loss: {:.4f} '.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_sal.data, loss_rgb.data, loss_depth.data, loss.data))
        writer.add_scalar('Loss/rgb', loss_rgb.item(), iteration)
        writer.add_scalar('Loss/depth', loss_depth.item(), iteration)
        writer.add_scalar('Loss/train', loss_sal.item(), iteration)
        writer.add_scalar('Loss/edge', loss_main_edge.item(), iteration)
        writer.add_images('Results/rgb', dets_rgb.sigmoid(), iteration)
        writer.add_images('Results/depth_map', depth_o, iteration)
        writer.add_images('Results/calibrated_depth', depth_calibrated, iteration)
        writer.add_images('Results/depth', dets_depth.sigmoid(), iteration)
        writer.add_images('Results/Pred', pred.sigmoid(), iteration)
        writer.add_images('Results/edge', edge_main3.sigmoid(), iteration)

    if opt.is_ResNet:
        save_path = 'ckpt/DCF_Resnet/'
    else:
        save_path = 'ckpt/DCF_VGG/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 2 == 0:
        torch.save(model_rgb.state_dict(), save_path + 'DCF_rgb.pth' + '.%d' % (epoch+1))
        torch.save(model_depth.state_dict(), save_path + 'DCF_depth.pth' + '.%d' % (epoch + 1))
        torch.save(model.state_dict(), save_path + 'DCF.pth' + '.%d' % (epoch + 1))
        torch.save(model_discriminator.state_dict(), save_path + 'DCF_dis.pth' + '.%d' % (epoch + 1))
        torch.save(model_estimator.state_dict(), save_path + 'DCF_estimator.pth' + '.%d' % (epoch + 1))


print("Let's go!")
for epoch in range(1, opt.epoch):
    adjust_lr(optimizer_rgb, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    adjust_lr(optimizer_depth, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model_rgb, model_depth, model,
          optimizer_rgb, optimizer_depth,optimizer, epoch)
    if (epoch+1) % 2 == 0:
        ckpt_name = '.' + str(epoch+1)
        eval_data(val_root, validation,ckpt_name)
    if epoch >= opt.epoch -1:
        writer.close()
