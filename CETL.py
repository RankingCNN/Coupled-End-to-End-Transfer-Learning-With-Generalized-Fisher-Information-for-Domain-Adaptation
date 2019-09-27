from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model.build_gen import *
from datasets.dataset_read import dataset_read
from PIL import Image
import torchvision
import numpy as np
from torchvision.utils import make_grid

# Training settings only need 5 epochs
class Cetl(object):
    def __init__(self, args, batch_size=64, source='svhn',
                 target='mnist', learning_rate=0.0002, interval=100, optimizer='adam'
                 , num_k=4, all_use=False, checkpoint_dir=None, save_epoch=10):
        self.batch_size = batch_size
        self.source = source
        self.target = target
        self.num_k = num_k
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff
        self.all_use = all_use
        if self.source == 'svhn':
            self.scale = True
        else:
            self.scale = False
        print('dataset loading')
        self.datasets, self.dataset_test = dataset_read(source, target, self.batch_size, scale=self.scale,
                                                        all_use=self.all_use)
        print('load finished!')
        self.G = Generator(source=source, target=target)
        self.C1 = Classifier(source=source, target=target)
        self.C2 = Classifier(source=source, target=target)
        self.D1 = D(source=source, target=target)
        self.D2 = D(source=source, target=target)
        if args.eval_only:
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, args.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (
                    self.checkpoint_dir, self.source, self.target, self.checkpoint_dir, args.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, args.resume_epoch))

        self.G.cuda()
        self.C1.cuda()
        self.C2.cuda()
        self.D1.cuda()
        self.D2.cuda()
        self.interval = interval

        self.set_optimizer(which_opt=optimizer, lr=learning_rate)
        self.lr = learning_rate

    def set_optimizer(self, which_opt='adam', lr=0.0001):       
        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),
                                    lr=lr, weight_decay=0.0005)
            self.opt_d1 = optim.Adam(self.D1.parameters(),
                                    lr=lr, weight_decay=0.0005)
            
            self.opt_d2 = optim.Adam(self.D2.parameters(),
                                     lr=lr, weight_decay=0.0005)

            self.opt_c1 = optim.Adam(self.C1.parameters(),
                                     lr=lr, weight_decay=0.0005)
            self.opt_c2 = optim.Adam(self.C2.parameters(),
                                     lr=lr, weight_decay=0.0005)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()
        self.opt_d1.zero_grad()
        self.opt_d2.zero_grad()

    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))
    
    
    def discrepancy2(self, out1, out2):
        out1=out1.view(out1.size(0), 3*32*32)
        out2=out2.view(out2.size(0), 3*32*32)
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))
        #return torch.mean(torch.abs(out1 - out2))
    
    
    def get_impulse_noise(self, X, level):
        p = 1. - level
        X = X.numpy()
        Y = X * np.random.binomial(1, p, size=X.shape)
        return Y
    
    def train(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        mse = nn.MSELoss().cuda()
        self.G.train()
        self.C1.train()
        self.C2.train()
        self.D1.train()
        self.D2.train()
        torch.cuda.manual_seed(1)
        
        for batch_idx, data in enumerate(self.datasets):
            img_t = data['T']
            img_s = data['S']
            label_s = data['S_label']
            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break
            img_tc = img_t.clone()
            img_t = self.get_impulse_noise(img_t, 0.5)
            img_t = torch.from_numpy(img_t)
            img_t = img_t.float().cuda()
            img_tc = img_tc.cuda()
            
            img_t = Variable(img_t)
            feat_t = self.G(img_t)
            recon1= self.D1(feat_t)
            recon2 = self.D2(feat_t)
            loss_t = mse(recon1, img_tc) + mse(recon2, img_tc)
            loss_t.backward()
            self.opt_g.step()
            self.opt_d1.step()
            self.opt_d2.step()
            self.reset_grad()
            
        for batch_idx, data in enumerate(self.datasets):
            img_t = data['T']
            img_s = data['S']
            
            label_s = data['S_label']
            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break
            img_tc = img_t.clone()
            #img_t = self.get_impulse_noise(img_t, 0.5)
            #img_t = torch.from_numpy(img_t)
            
            
            img_s = img_s.cuda()
            #img_t = img_t.float().cuda()
            img_t = img_t.cuda()
            img_tc = img_tc.cuda()
            
            
            label_s = Variable(label_s.long().cuda())
            
            
            img_s = Variable(img_s)
            img_t = Variable(img_t)
            self.reset_grad()
            feat_s = self.G(img_s)
            feat_t = self.G(img_t)
            
            
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
            
            
            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            
            loss_s = loss_s1 + loss_s2
            loss_s.backward()
            self.opt_g.step()
            self.opt_d1.step()
            self.opt_d2.step()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()
            
            #calculate
            feat_s = self.G(img_s)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
            
            feat_t = self.G(img_t)
            output_t1 = self.C1(feat_t)
            output_t2 = self.C2(feat_t)
            
            
            feat_t = self.G(img_t)
            
            re_s1 = self.D1(feat_s)
            re_s2 = self.D2(feat_s)
            
            
            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            
            recon1 = self.D1(feat_t)
            recon2 = self.D2(feat_t)
            loss_r1 = mse(recon2, img_tc)
            loss_r2 = mse(recon1, img_tc)
            loss_s = loss_s1 + loss_s2 + loss_r1 + loss_r2
            loss_dis = self.discrepancy(output_t1, output_t2)+self.discrepancy2(re_s1, re_s2)
            loss = loss_s - loss_dis
            loss.backward()
            self.opt_c1.step()
            self.opt_c2.step()
            self.opt_d1.step()
            self.opt_d2.step()
            self.reset_grad()

            for i in range(self.num_k):
                #sorce
                feat_s = self.G(img_s)
                re_s1 = self.D1(feat_s)
                re_s2 = self.D2(feat_s)
                #target
                feat_t = self.G(img_t)
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)
                
                loss_dis = self.discrepancy(output_t1, output_t2)+self.discrepancy2(re_s1, re_s2)
                loss_dis.backward()
                self.opt_g.step()
                self.reset_grad()
                
            
            if batch_idx > 500:
                return batch_idx

            if batch_idx % self.interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t cls1: {:.6f}\t cls2: {:.6f} \t Discrepancy: {:.6f}\t lossRt1: {:.6f}'.format(
                    epoch, batch_idx, 100,
                    100. * batch_idx / 70000, loss_s1.data[0], loss_s2.data[0], loss_dis.data[0],loss_r1.data[0]))
                
                torchvision.utils.save_image(make_grid(recon1[0:16,:,:,:], nrow=4), '../saveImg/re_t.jpg')
                torchvision.utils.save_image(make_grid(re_s2[0:16,:,:,:], nrow=4), '../saveImg/re_s2.jpg')
                '''
                if record_file:
                    record = open(record_file, 'a')
                    record.write('%s %s %s\n' % (loss_dis.data[0], loss_s1.data[0], loss_s2.data[0]))
                    record.close()
                '''
        return batch_idx

    

    def test(self, epoch, record_file=None, save_model=False):
        self.G.eval()
        self.C1.eval()
        self.C2.eval()
        test_loss = 0
        correct1 = 0
        correct2 = 0
        correct3 = 0
        size = 0
        for batch_idx, data in enumerate(self.dataset_test):
            img = data['T']
            label = data['T_label']
            img, label = img.cuda(), label.long().cuda()
            img, label = Variable(img, volatile=True), Variable(label)
            feat = self.G(img)
            output1 = self.C1(feat)
            output2 = self.C2(feat)
            test_loss += F.nll_loss(output1, label).data[0]
            output_ensemble = output1 + output2
            pred1 = output1.data.max(1)[1]
            pred2 = output2.data.max(1)[1]
            pred_ensemble = output_ensemble.data.max(1)[1]
            k = label.data.size()[0]
            correct1 += pred1.eq(label.data).cpu().sum()
            correct2 += pred2.eq(label.data).cpu().sum()
            correct3 += pred_ensemble.eq(label.data).cpu().sum()
            size += k
        test_loss = test_loss / size
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%) Accuracy C2: {}/{} ({:.0f}%) Accuracy Ensemble: {}/{} ({:.0f}%) \n'.format(
                test_loss, correct1, size,
                100. * correct1 / size, correct2, size, 100. * correct2 / size, correct3, size, 100. * correct3 / size))
        if save_model and epoch % self.save_epoch == 0:
            torch.save(self.G,
                       '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
            torch.save(self.C1,
                       '%s/%s_to_%s_model_epoch%s_C1.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
            torch.save(self.C2,
                       '%s/%s_to_%s_model_epoch%s_C2.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
        '''
        if record_file:
            record = open(record_file, 'a')
            #print('recording %s', record_file)
            record.write('%s %s %s\n' % (float(correct1) / size, float(correct2) / size, float(correct3) / size))
            record.close()
        '''