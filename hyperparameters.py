#----------------------------------------------------
# Hyperparameters definition
# Author: Pablo Villanueva Domingo
# Last update: 4/22
#----------------------------------------------------

# Hyperparameters class
class hyperparameters():
    def __init__(self, learning_rate,flow_hidden_features = 128, flow_num_transforms = 5, flow_training_batch_size = 10, save=True):

        # Learning rate
        self.learning_rate = learning_rate
        # flow hparams        
        self.flow_hidden_features = flow_hidden_features
        self.flow_num_transforms = flow_num_transforms
        self.flow_training_batch_size = flow_training_batch_size
        #whether to save model
        self.save=save

    # Name of the model and hyperparameters
    def name_model(self):
        return "lr_{:.2e}_features_{:d}_transforms_{:d}_batchsize_{:d}".format(self.learning_rate, self.flow_hidden_features, self.flow_num_transforms, self.flow_training_batch_size)


#--- HYPERPARAMETER CHOICES ---#

hparams = hyperparameters(learning_rate = 1.e-04,    # Learning rate
                          flow_hidden_features = 64,
                          flow_num_transforms = 3,
                          flow_training_batch_size = 32,
                          save = True
                          )


