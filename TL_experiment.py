from train import train
from test import test
import csv
from remove_word import remove

def TL_experiment():
    train_source_feature_name_list = ['train_X_kla240.mat', 'train_X_kla240.mat', 
                                      'train_X_kla360.mat', 'train_X_kla360.mat', 
                                      'train_X_kla480.mat', 'train_X_kla480.mat']
    train_source_label_name_list = ['train_EQvec_kla240.mat', 'train_EQvec_kla240.mat', 
                                    'train_EQvec_kla360.mat', 'train_EQvec_kla360.mat', 
                                    'train_EQvec_kla480.mat', 'train_EQvec_kla480.mat']
    train_target_feature_name_list = ['train_X_mu0.7.mat', 'train_X_mu0.9.mat',
                                      'train_X_mu0.7.mat', 'train_X_mu0.9.mat',
                                      'train_X_mu0.7.mat', 'train_X_mu0.9.mat']
    train_target_label_name_list = ['train_EQvec_mu0.7.mat', 'train_EQvec_mu0.9.mat',
                                    'train_EQvec_mu0.7.mat', 'train_EQvec_mu0.9.mat',
                                    'train_EQvec_mu0.7.mat', 'train_EQvec_mu0.9.mat']
    
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
        
    nums_experiments = len(train_source_feature_name_list)
    # source_r2_result_list = []
    # source_rmse_result_list = []
    # source_label_loss_list = []
    # target_r2_result_list = []
    # target_rmse_result_list = []
    # target_label_loss_list = []
    
    for i in range(nums_experiments):
        train_source_dataset_name = train_source_feature_name_list[i]
        train_target_dataset_name = train_target_feature_name_list[i]
        train_source_label_name = train_source_label_name_list[i]
        train_target_label_name = train_target_label_name_list[i]
        
        test_source_dataset_name = test_source_feature_name_list[i]
        test_target_dataset_name = test_target_feature_name_list[i]
        test_source_label_name = test_source_label_name_list[i]
        test_target_label_name = test_target_label_name_list[i]
        
        # train
        train(source_feature=train_source_dataset_name,
            source_label=train_source_label_name,
            target_feature=train_target_dataset_name,
            target_label=train_target_label_name)
        
        # test
        source_r2, source_RMSE, target_r2, target_RMSE = test(source_feature=test_source_dataset_name,
                                                            source_label=test_source_label_name,
                                                            target_feature=test_target_dataset_name,
                                                            target_label=test_target_label_name)
        
        # write the results to the .csv file in process_data folder with the column name
        with open('process_data/results.csv', mode='a') as results_file:
            results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_writer.writerow(['source_dataset_name', 'target_dataset_name', 'source_r2', 'source_RMSE', 'target_r2', 'target_RMSE'])
            results_writer.writerow([remove(train_source_dataset_name), remove(train_target_dataset_name), source_r2, source_RMSE,target_r2, target_RMSE])

if __name__ == "__main__":
    TL_experiment()
    print("done")

