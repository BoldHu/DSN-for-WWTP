import torch.nn as nn
from functions import ReverseLayerF


class DSN(nn.Module):
    def __init__(self):
        super(DSN, self).__init__()

        ##########################################
        # private source encoder
        ##########################################

        self.source_encoder = nn.Sequential()
        self.source_encoder.add_module('ps_linear1', nn.Linear(15, 64))
        self.source_encoder.add_module('ps_LeakyReLU1', nn.LeakyReLU(True))
        self.source_encoder.add_module('ps_linear2', nn.Linear(64, 128))
        self.source_encoder.add_module('ps_LeakyReLU2', nn.LeakyReLU(True))
        self.source_encoder.add_module('ps_linear3', nn.Linear(128, 64))
        self.source_encoder.add_module('ps_LeakyReLU3', nn.LeakyReLU(True))
        # self.source_encoder.add_module('ps_linear3', nn.Linear(128, 256))
        # self.source_encoder.add_module('ps_LeakyReLU3', nn.LeakyReLU(True))
        # self.source_encoder.add_module('ps_linear4', nn.Linear(256, 128))
        # self.source_encoder.add_module('ps_LeakyReLU4', nn.LeakyReLU(True))
        # self.source_encoder.add_module('ps_linear5', nn.Linear(128, 64))
        # self.source_encoder.add_module('ps_LeakyReLU5', nn.LeakyReLU(True))
        
        #########################################
        # private target encoder
        #########################################

        self.target_encoder = nn.Sequential()
        self.target_encoder.add_module('pt_linear1', nn.Linear(15, 64))
        self.target_encoder.add_module('pt_LeakyReLU1', nn.LeakyReLU(True))
        self.target_encoder.add_module('pt_linear2', nn.Linear(64, 128))
        self.target_encoder.add_module('pt_LeakyReLU2', nn.LeakyReLU(True))
        self.target_encoder.add_module('pt_linear3', nn.Linear(128, 64))
        self.target_encoder.add_module('pt_LeakyReLU3', nn.LeakyReLU(True))         
        # self.target_encoder.add_module('pt_linear3', nn.Linear(128, 256))
        # self.target_encoder.add_module('pt_LeakyReLU3', nn.LeakyReLU(True))        
        # self.target_encoder.add_module('pt_linear4', nn.Linear(256, 128))
        # self.target_encoder.add_module('pt_LeakyReLU4', nn.LeakyReLU(True))
        # self.target_encoder.add_module('pt_linear5', nn.Linear(128, 64))
        # self.target_encoder.add_module('pt_LeakyReLU5', nn.LeakyReLU(True))
                
        ################################
        # shared encoder (dann_mnist)
        ################################

        self.shared_encoder = nn.Sequential()
        self.shared_encoder.add_module('sc_linear1', nn.Linear(15, 64))
        self.shared_encoder.add_module('sc_LeakyReLU1', nn.LeakyReLU(True))
        self.shared_encoder.add_module('sc_linear2', nn.Linear(64, 128))
        self.shared_encoder.add_module('sc_LeakyReLU2', nn.LeakyReLU(True))
        self.shared_encoder.add_module('sc_linear3', nn.Linear(128, 64))
        self.shared_encoder.add_module('sc_LeakyReLU3', nn.LeakyReLU(True))        
        # self.shared_encoder.add_module('sc_linear3', nn.Linear(128, 256))
        # self.shared_encoder.add_module('sc_LeakyReLU3', nn.LeakyReLU(True))
        # self.shared_encoder.add_module('sc_linear4', nn.Linear(256, 128))
        # self.shared_encoder.add_module('sc_LeakyReLU4', nn.LeakyReLU(True))
        # self.shared_encoder.add_module('sc_linear5', nn.Linear(128, 64))
        # self.shared_encoder.add_module('sc_LeakyReLU5', nn.LeakyReLU(True))

        # predict the regression value
        self.shared_encoder_pred_reg = nn.Sequential()
        self.shared_encoder_pred_reg.add_module('pr_linear1', nn.Linear(64, 1))

        # predict the domain
        self.shared_encoder_pred_domain = nn.Sequential()
        self.shared_encoder_pred_domain.add_module('pd_linear1', nn.Linear(64, 64))
        self.shared_encoder_pred_domain.add_module('pd_LeakyReLU1', nn.LeakyReLU(True))
        self.shared_encoder_pred_domain.add_module('pd_linear2', nn.Linear(64, 32))
        self.shared_encoder_pred_domain.add_module('pd_LeakyReLU2', nn.LeakyReLU(True))
        self.shared_encoder_pred_domain.add_module('pd_linear3', nn.Linear(32, 2))
        
        ######################################
        # shared decoder (small decoder)
        ######################################

        self.shared_decoder = nn.Sequential()
        self.shared_decoder.add_module('sd_linear1', nn.Linear(64, 64))
        self.shared_decoder.add_module('sd_LeakyReLU1', nn.LeakyReLU(True))
        self.shared_decoder.add_module('sd_linear2', nn.Linear(64, 32))
        self.shared_decoder.add_module('sd_LeakyReLU2', nn.LeakyReLU(True))
        self.shared_decoder.add_module('sd_linear3', nn.Linear(32, 15))
        
    def forward(self, input_data, mode, rec_scheme, p=0.0):

        result = []

        if mode == 'source':
            # source private encoder
            private_code = self.source_encoder(input_data)

        elif mode == 'target':
            # target private encoder
            private_code = self.target_encoder(input_data)

        result.append(private_code)

        # shared encoder
        shared_code = self.shared_encoder(input_data)
        result.append(shared_code)

        reversed_shared_code = ReverseLayerF.apply(shared_code, p)
        domain_label = self.shared_encoder_pred_domain(reversed_shared_code)
        result.append(domain_label)

        if mode == 'source':
            reg_label = self.shared_encoder_pred_reg(shared_code)
            result.append(reg_label)

        # shared decoder

        if rec_scheme == 'share':
            union_code = shared_code
        elif rec_scheme == 'all':
            union_code = private_code + shared_code
        elif rec_scheme == 'private':
            union_code = private_code

        rec_code = self.shared_decoder(union_code)
        result.append(rec_code)

        return result