# hyperparameters are defined in this class


class hyperparameters():
    def __init__(self, learning_rate,flow_hidden_features = 128, flow_num_transforms = 5, flow_training_batch_size = 8, 
                 in_channels = 4, num_filts = 1, filter_size = 35, order = 2, save=True, device='cpu'):

        # Learning rate
        self.learning_rate = learning_rate
        # flow hparams        
        self.flow_hidden_features = flow_hidden_features
        self.flow_num_transforms = flow_num_transforms
        self.flow_training_batch_size = flow_training_batch_size
        # C3NN hparams
        self.in_channels = in_channels
        self.num_filts = num_filts
        self.filter_size = filter_size
        self.order = order


        #whether to save model
        self.save=save

        #device to use
        self.device=device

    # Name of the model and hyperparameters
    def name_model(self):
        # TODO: include c3nn params
        return "lr_{:.2e}_features_{:d}_transforms_{:d}_batchsize_{:d}".format(self.learning_rate, self.flow_hidden_features, self.flow_num_transforms, self.flow_training_batch_size)


#--- HYPERPARAMETER CHOICES ---#

hparams = hyperparameters(learning_rate = 1.e-04,    # Learning rate
                          flow_hidden_features = 64,
                          flow_num_transforms = 3,
                          flow_training_batch_size = 32,
                          in_channels = 4, # not optimzed
                          num_filts = 1, # The number of filters to learn in our model
                          filter_size = 35, # The spatial length of the filters
                          order = 2, # The highest order of correlation measured, not optimized
                          save = True,  #TODO: not yet implemented
                          device = 'cpu'
                          )


