import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from model_compat import DSN
from data_loader import GetLoader
from functions import SIMSE, DiffLoss, MSE
from test import test

######################
# params             #
######################

source_dataset_name = 'X_kla120.mat'
target_dataset_name = 'X_kla240.mat'
source_dataset_labels_name = 'EQvec_kla120.mat'
target_dataset_labels_name = 'EQvec_kla240.mat'
model_root = 'models'
cuda = True
cudnn.benchmark = True
lr = 1e-2
batch_size = 32
n_epoch = 100
step_decay_weight = 0.95
lr_decay_step = 20000
active_domain_loss_step = 10000
weight_decay = 1e-6
alpha_weight = 0.01
beta_weight = 0.075
gamma_weight = 0.25
momentum = 0.9

random.seed(42)
torch.manual_seed(42)

#######################
# load data           #
#######################

source_dataset = GetLoader(source_dataset_name, source_dataset_labels_name, transform=True)
target_dataset = GetLoader(target_dataset_name, target_dataset_labels_name, transform=True)

# create dataloaders
dataloader_source = torch.utils.data.DataLoader(
    dataset=source_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=32)

dataloader_target = torch.utils.data.DataLoader(
    dataset=target_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=32)

print('read the data from the dataset')

#####################
#  load model       #
#####################

my_net = DSN()

#####################
# setup optimizer   #
#####################


def exp_lr_scheduler(optimizer, step, init_lr=lr, lr_decay_step=lr_decay_step, step_decay_weight=step_decay_weight):

    # Decay learning rate by a factor of step_decay_weight every lr_decay_step
    current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))

    if step % lr_decay_step == 0:
        print ('learning rate is set to %f' % current_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return optimizer


optimizer = optim.SGD(my_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

loss_reg = torch.nn.MSELoss()
loss_recon1 = MSE()
loss_recon2 = SIMSE()
loss_diff = DiffLoss()
loss_similarity = torch.nn.CrossEntropyLoss()

if cuda:
    my_net = my_net.cuda()
    loss_reg = loss_reg.cuda()
    loss_recon1 = loss_recon1.cuda()
    loss_recon2 = loss_recon2.cuda()
    loss_diff = loss_diff.cuda()
    loss_similarity = loss_similarity.cuda()

for p in my_net.parameters():
    p.requires_grad = True

#############################
# training network          #
#############################


len_dataloader = min(len(dataloader_source), len(dataloader_target))
dann_epoch = np.floor(active_domain_loss_step / len_dataloader * 1.0)

current_step = 0
for epoch in range(n_epoch):

    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0

    while i < len_dataloader:

        ###################################
        # target data training            #
        ###################################

        data_target = data_target_iter.next()
        target_feature, target_label = data_target

        my_net.zero_grad()
        loss = 0
        batch_size = len(target_label)

        domain_label = torch.ones(batch_size)
        domain_label = domain_label.long()

        if cuda:
            target_feature = target_feature.cuda()
            target_label = target_label.cuda()
            domain_label = domain_label.cuda()

        if current_step > active_domain_loss_step:
            p = float(i + (epoch - dann_epoch) * len_dataloader / (n_epoch - dann_epoch) / len_dataloader)
            p = 2. / (1. + np.exp(-10 * p)) - 1

            # activate domain loss
            result = my_net(input_data=target_feature, mode='target', rec_scheme='all', p=p)
            target_privte_code, target_share_code, target_domain_label, target_rec_code = result
            target_dann = gamma_weight * loss_similarity(target_domain_label, domain_label)
            loss += target_dann
        else:
            target_dann = Variable(torch.zeros(1).float().cuda())
            result = my_net(input_data=target_feature, mode='target', rec_scheme='all')
            target_privte_code, target_share_code, _, target_rec_code = result

        target_diff= beta_weight * loss_diff(target_privte_code, target_share_code)
        loss += target_diff
        target_mse = alpha_weight * loss_recon1(target_rec_code, target_feature)
        loss += target_mse
        target_simse = alpha_weight * loss_recon2(target_rec_code, target_feature)
        loss += target_simse

        loss.backward()
        optimizer.step()

        ###################################
        # source data training            #
        ###################################

        data_source = data_source_iter.next()
        source_feature, source_label = data_source

        my_net.zero_grad()
        batch_size = len(source_label)
        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()

        loss = 0

        if cuda:
            source_feature = source_feature.cuda()
            source_label = source_label.cuda()
            domain_label = domain_label.cuda()

        if current_step > active_domain_loss_step:

            # activate domain loss

            result = my_net(input_data=source_feature, mode='source', rec_scheme='all', p=p)
            source_privte_code, source_share_code, source_domain_label, source_class_label, source_rec_code = result
            source_dann = gamma_weight * loss_similarity(source_domain_label, domain_label)
            loss += source_dann
        else:
            source_dann = Variable(torch.zeros(1).float().cuda())
            result = my_net(input_data=source_feature, mode='source', rec_scheme='all')
            source_privte_code, source_share_code, _, source_reg, source_rec_code = result

        source_classification = loss_reg(source_reg, source_label)
        loss += source_classification

        source_diff = beta_weight * loss_diff(source_privte_code, source_share_code)
        loss += source_diff
        source_mse = alpha_weight * loss_recon1(source_rec_code, source_feature)
        loss += source_mse
        source_simse = alpha_weight * loss_recon2(source_rec_code, source_feature)
        loss += source_simse

        loss.backward()
        optimizer = exp_lr_scheduler(optimizer=optimizer, step=current_step)
        optimizer.step()

        i += 1
        current_step += 1
    print ('source_classification: %f, source_dann: %f, source_diff: %f, ' \
          'source_mse: %f, source_simse: %f, target_dann: %f, target_diff: %f, ' \
          'target_mse: %f, target_simse: %f' \
          % (source_classification.data.cpu().numpy(), source_dann.data.cpu().numpy(), source_diff.data.cpu().numpy(),
             source_mse.data.cpu().numpy(), source_simse.data.cpu().numpy(), target_dann.data.cpu().numpy(),
             target_diff.data.cpu().numpy(),target_mse.data.cpu().numpy(), target_simse.data.cpu().numpy()))

    # print 'step: %d, loss: %f' % (current_step, loss.cpu().data.numpy())
    torch.save(my_net.state_dict(), model_root + '/dsn_mnist_mnistm_epoch_' + str(epoch) + '.pth')
    test(epoch=epoch, name='mnist')
    test(epoch=epoch, name='mnist_m')

print ('done')





