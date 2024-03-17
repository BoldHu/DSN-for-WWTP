import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from model_compat import DSN
from data_loader import GetLoader
from functions import SIMSE, DiffLoss, MSE

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
lr = 1e-10
batch_size = 64
n_epoch = 100
step_decay_weight = 0.9
lr_decay_step = 10
active_domain_loss_step = 0
weight_decay = 1e-7
alpha_weight = 0.001
beta_weight = 0.01
gamma_weight = 0.1
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

        data_target = next(data_target_iter)
        target_feature, target_label = data_target
        print(target_feature.shape)
        print(target_label.shape)

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
            # print(result[0])
            # print(result[0].shape)
            # print(result[1])
            # print(result[1].shape)
            # print(result[2])
            # print(result[2].shape)
            # print(result[3])
            # print(result[3].shape)
            target_privte_code, target_share_code, target_domain_label, target_rec_code = result
            target_dann = gamma_weight * loss_similarity(target_domain_label, domain_label)
            loss += target_dann
        else:
            target_dann = torch.zeros(1).float().cuda()
            result = my_net(input_data=target_feature, mode='target', rec_scheme='all')
            # print(result[0])
            # print(result[0].shape)
            # print(result[1])
            # print(result[1].shape)
            # print(result[2])
            # print(result[2].shape)
            # print(result[3])
            # print(result[3].shape)
            target_privte_code, target_share_code, _, target_rec_code = result

        target_diff= beta_weight * loss_diff(target_privte_code, target_share_code)
        print(target_diff)
        loss += target_diff
        target_mse = alpha_weight * loss_recon1(target_rec_code, target_feature)
        print(target_mse)
        loss += target_mse
        target_simse = alpha_weight * loss_recon2(target_rec_code, target_feature)
        print(target_simse)
        loss += target_simse

        loss.backward()
        optimizer.step()

        ###################################
        # source data training            #
        ###################################

        data_source = next(data_source_iter)
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
            source_privte_code, source_share_code, source_domain_label, source_reg, source_rec_code = result
            source_dann = gamma_weight * loss_similarity(source_domain_label, domain_label)
            loss += source_dann
        else:
            source_dann = torch.zeros(1).float().cuda()
            result = my_net(input_data=source_feature, mode='source', rec_scheme='all')
            source_privte_code, source_share_code, _, source_reg, source_rec_code = result

        source_regression_loss = loss_reg(source_reg, source_label)
        print(source_regression_loss)
        loss += source_regression_loss

        source_diff = beta_weight * loss_diff(source_privte_code, source_share_code)
        print(source_diff)
        loss += source_diff
        source_mse = alpha_weight * loss_recon1(source_rec_code, source_feature)
        print(source_mse)
        loss += source_mse
        source_simse = alpha_weight * loss_recon2(source_rec_code, source_feature)
        print(source_simse)
        loss += source_simse

        loss.backward()
        optimizer = exp_lr_scheduler(optimizer=optimizer, step=current_step)
        optimizer.step()

        i += 1
        current_step += 1
        
    print('epoch: %d, loss: %f, source_regression_loss: %f, source_diff: %f, source_mse: %f, source_simse: %f, target_dann: %f, target_diff: %f, target_mse: %f, target_simse: %f' %
          (epoch, loss.item(), source_regression_loss.item(), source_diff.item(), source_mse.item(), source_simse.item(), target_dann.item(), target_diff.item(), target_mse.item(), target_simse.item()))
    print(loss.item())
    torch.save(my_net.state_dict(), model_root + '/dsn_epoch_' + str(epoch) + '.pth')

print ('done')