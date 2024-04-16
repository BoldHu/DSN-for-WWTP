import os
import torch
import torch.utils.data
from data_loader import GetLoader
from remove_word import remove
import matplotlib.pyplot as plt

def plot_scatter(source_feature, source_label, target_feature, target_label, task_number, source_kla, source_mu, target_kla, target_mu):
    batch_size = 1344
    alpha = 0
    
    # read the data from the dataset
    source_dataset_name = source_feature
    source_dataset_labels_name = source_label
    target_dataset_name = target_feature
    target_dataset_labels_name = target_label
        
    target_dataset = GetLoader(target_dataset_name, target_dataset_labels_name, transform=True)
    dataloader_target = torch.utils.data.DataLoader(
        dataset=target_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=32)
    
    source_dataset = GetLoader(source_dataset_name, source_dataset_labels_name, transform=True)
    dataloader_source = torch.utils.data.DataLoader(
        dataset=source_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=32)
    
    len_dataloader = min(len(dataloader_target), len(dataloader_source))
        
    # load model in saved_models
    source_dataset_name = remove(source_dataset_name)
    target_dataset_name = remove(target_dataset_name)
    my_net = torch.load(os.path.join('models', source_dataset_name + '_' + target_dataset_name + '.pth'))
    
    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    my_net = my_net.to(device)
    
    source_outputs = []
    source_labels = []
    target_outputs = []
    target_labels = []
    
    with torch.no_grad():
        for test_features, test_labels in dataloader_target:
            test_features = test_features.to(device)
            output, _ = my_net(input_data=test_features, alpha=alpha)
            target_outputs.extend(output.cpu().numpy())
            target_labels.extend(test_labels.cpu().numpy())

        for test_features, test_labels in dataloader_source:
            test_features = test_features.to(device)
            output, _ = my_net(input_data=test_features, alpha=alpha)
            source_outputs.extend(output.cpu().numpy())
            source_labels.extend(test_labels.cpu().numpy())
    
    # plot scatter with source_output, source_label and target_output, target_label
    plt.figure(figsize=(7, 5))
    plt.scatter(source_outputs, source_labels, color='blue', alpha=0.5, label='Source data')
    plt.scatter(target_outputs, target_labels, color='red', alpha=0.5, label='Target data')
    # plt.title(r'Task{0} Transfer From $L_{l}a={1}$ and $\mu={2}$ to $L_{l}a={3}$ and $\mu={4}$'.format(task_number, source_kla, source_mu, target_kla, target_mu))
    plt.title(r'Task{0} Transfer From $L_{{l}}a={1}$ and $\mu={2}$ to $L_{{l}}a={3}$ and $\mu={4}$'.format(task_number, source_kla, source_mu, target_kla, target_mu))
    plt.xlabel('Predicted EQ')
    plt.ylabel('True EQ')
    plt.legend()

    # Add y=x line to compare predictions to true values
    min_val = min(min(source_outputs), min(target_outputs), min(source_labels), min(target_labels))
    max_val = max(max(source_outputs), max(target_outputs), max(source_labels), max(target_labels))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x (Ideal)')
    plt.legend()

    # save the plot with souce and target dataset names
    plt.savefig(os.path.join('figures', 'scatter_plot_' + source_dataset_name + '_' + target_dataset_name + '.png'))
    plt.show()

if __name__ == "__main__":
    task_list = ['1', '2', '3', '4', '5', '6']
    source_kla_list = ['240', '240', '360', '360', '480', '480']
    source_mu_list = ['0.5', '0.5', '0.5', '0.5', '0.5', '0.5']
    target_kla_list = ['0.7', '0.9', '0.7', '0.9', '0.7', '0.9']
    target_mu_list = ['240', '240', '240', '240', '240', '240']
    test_source_feature_name_list = ['test_X_kla240.mat', 'test_X_kla240.mat', 
                                     'test_X_kla360.mat', 'test_X_kla360.mat', 
                                     'test_X_kla480.mat', 'test_X_kla480.mat']
    test_source_label_name_list = ['test_EQvec_kla240.mat', 'test_EQvec_kla240.mat', 
                                   'test_EQvec_kla360.mat', 'test_EQvec_kla360.mat', 
                                   'test_EQvec_kla480.mat', 'test_EQvec_kla480.mat']
    test_target_feature_name_list = ['test_X_mu0.7.mat', 'test_X_mu0.9.mat',
                                     'test_X_mu0.7.mat', 'test_X_mu0.9.mat',
                                     'test_X_mu0.7.mat', 'test_X_mu0.9.mat']
    test_target_label_name_list = ['test_EQvec_mu0.7.mat', 'test_EQvec_mu0.9.mat',
                                   'test_EQvec_mu0.7.mat', 'test_EQvec_mu0.9.mat',
                                   'test_EQvec_mu0.7.mat', 'test_EQvec_mu0.9.mat']
    for j in range(len(test_source_feature_name_list)):
        plot_scatter(source_feature=test_source_feature_name_list[j], 
                        source_label=test_source_label_name_list[j], 
                        target_feature=test_target_feature_name_list[j], 
                        target_label=test_target_label_name_list[j],
                        task_number=task_list[j],
                        source_kla=source_kla_list[j],
                        source_mu=source_mu_list[j],
                        target_kla=target_kla_list[j],
                        target_mu=target_mu_list[j])


