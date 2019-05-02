from keras.models import Model as KerasModel
from keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import pandas as pd
import numpy as np
from models.model import Model

class NeuralNet(Model):
    def __init__(self, model_params):
        super().__init__(model_params,'Neural Network Classifier')
    
    def build_model(self):
        num_hidden_layers = self.model_params['num_hidden_layers']
        nodes_per_layer = self.model_params['nodes_per_layer']
        activation = self.model_params['activation']
        optimizer = self.model_params['optimizer']
        metrics = self.model_params['metrics']
        num_features = self.model_params['num_features']
        num_targets = self.model_params['num_targets']

        if self.model_params['loss']=='crossentropy':
            if num_targets>1:
                loss='categorical_crossentropy'
            else:
                loss='binary_crossentropy'
        else:
            loss = self.model_params['loss']

        input_layer = Input(shape=(num_features,))

        # Input Layer is not actually a hidden layer, this is just used
        # for ease of coding
        hidden_layer_i = input_layer
        for i in range(num_hidden_layers):
            hidden_layer_i = Dense(nodes_per_layer,activation=activation)(hidden_layer_i)
        output_layer = Dense(num_targets,activation='softmax')(hidden_layer_i)

        nn = KerasModel(inputs=input_layer,outputs=output_layer)
        nn.compile(optimizer=optimizer,loss=loss,metrics=metrics)
        
        return nn
    
    def fit(self, model, x_train, y_train,verbose=False):
        num_epochs = self.model_params['num_epochs']
        class_weight={}
        num_samples = y_train.shape[0]
        for i in range(y_train.shape[1]):
            num_i = np.sum(y_train[:,i]==1)
            ratio = num_i/num_samples
            if ratio!=0:
                class_weight[i]=1/ratio
            else:
                class_weight[i]=1
        x_train = StandardScaler().fit_transform(x_train)
        model.fit(x_train,y_train,epochs=num_epochs,class_weight=class_weight,verbose=verbose)
        return model
    
    def predict(self,model,x):
        x = StandardScaler().fit_transform(x)
        y_hat = model.predict(x)
        return y_hat
