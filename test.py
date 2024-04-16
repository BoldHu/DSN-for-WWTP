import os
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from data_loader import GetLoader
from model_compat import DSN
from reg_functions import reg_indicator
from remove_word import remove

def test(source_feature, source_label, target_feature, target_label, model=None):
    ###################
    # params          #
    ###################
    cuda = True
    cudnn.benchmark = True
    batch_size = 1344  # Adjust the batch size if necessary

    ###################
    # load data       #
    ###################

    source_dataset = GetLoader(source_feature, source_label, transform=True)
    target_dataset = GetLoader(target_feature, target_label, transform=True)
    
    source_dataloader = torch.utils.data.DataLoader(
        dataset=source_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for testing
        num_workers=4)  # Adjust according to your system capabilities

    target_dataloader = torch.utils.data.DataLoader(
        dataset=target_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    ####################
    # load model       #
    ####################
    
    if model is None:
        source_clean_name = remove(source_feature)
        target_clean_name = remove(target_feature)
        model_path = os.path.join('models', f'DSN_model_{source_clean_name}_{target_clean_name}.pth')
        model = torch.load(model_path)
    else:
        model = model
        
    if cuda:
        model = model.cuda()

    ####################
    # testing          #
    ####################
    
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        source_r2, source_rmse, target_r2, target_rmse = [], [], [], []

        for data in source_dataloader:
            source_features, source_labels = data
            if cuda:
                source_features = source_features.cuda()
                source_labels = source_labels.cuda()

            source_outputs = model(source_features, 'source', 'share', p=0)
            source_reg = source_outputs[-1]  # Assuming reg output is the last in the list

            r2, rmse = reg_indicator(source_labels, source_reg)
            source_r2.append(r2.item())
            source_rmse.append(rmse.item())

        for data in target_dataloader:
            target_features, target_labels = data
            if cuda:
                target_features = target_features.cuda()
                target_labels = target_labels.cuda()

            target_outputs = model(target_features, 'target', 'share', p=0)
            target_reg = target_outputs[-1]  # Assuming reg output is the last in the list

            r2, rmse = reg_indicator(target_labels, target_reg)
            target_r2.append(r2.item())
            target_rmse.append(rmse.item())
            
        source_r2_sum = sum(source_r2)
        source_rmse_sum = sum(source_rmse)
        target_r2_sum = sum(target_r2)
        target_rmse_sum = sum(target_rmse)

    return source_r2_sum / len(source_dataloader), source_rmse_sum / len(source_dataloader), target_r2_sum / len(target_dataloader), target_rmse_sum / len(target_dataloader)

# Example usage
if __name__ == "__main__":
    results = test('test_X_kla240.mat', 'test_EQvec_kla240.mat', 'test_X_mu0.7.mat', 'test_EQvec_mu0.7.mat')
    print(results)
