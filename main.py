from train import train
from test import test

train_source_feature_name = 'train_X_kla240.mat'
train_source_label_name = 'train_EQvec_kla240.mat'
train_target_feature_name = 'train_X_kla360.mat'
train_target_label_name = 'train_EQvec_kla360.mat'

test_source_feature_name = 'test_X_kla240.mat'
test_source_label_name = 'test_EQvec_kla240.mat'
test_target_feature_name = 'test_X_kla360.mat'
test_target_label_name = 'test_EQvec_kla360.mat'

source_r2, source_RMSE, source_label_loss = train(source_feature=train_source_feature_name,
                                                    source_label=train_source_label_name,
                                                    target_feature=train_target_feature_name,
                                                    target_label=train_target_label_name)

tarhet_r2, target_RMSE, target_label_loss = test(source_feature=test_source_feature_name,
                                                    source_label=test_source_label_name,
                                                    target_feature=test_target_feature_name,
                                                    target_label=test_target_label_name)

