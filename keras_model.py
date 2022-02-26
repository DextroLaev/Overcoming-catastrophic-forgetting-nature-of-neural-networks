import tensorflow as tf
import numpy as np
from dataset import Dataset
import matplotlib.pyplot as plt

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.accA=[]
        self.accB=[]
    
    def on_epoch_end(self,epoch,logs=None):
        self.accB.append(logs['accuracy'])
        self.accA.append(logs['val_accuracy'])
        if logs['accuracy']==1.0:
            self.model.stop_training=1  

    def on_train_end(self,logs=None):
        plt.plot(self.accB,label='accuracy')
        plt.plot(self.accA,label='val_accuracy')
        plt.legend()
        plt.show()           

class Continual_Learning_Neural_net:
    def __init__(self,lambda_val):
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.lambda_val = lambda_val
        self.model = self.model_arch()
        self.theta = []
        self.theta_star = []

    def model_arch(self):
        inputs = tf.keras.Input(shape=(28,28,1))
        l1 = tf.keras.layers.Flatten()(inputs)
        l2  = tf.keras.layers.Dense(10,activation='softmax')(l1)
        model = tf.keras.Model(inputs=inputs,outputs=l2)
        model.compile(loss=self.loss_fn,optimizer='adam',metrics=['accuracy'])
        return model

    def update_weights(self):
        self.theta = self.model.weights
        self.theta_star = self.model.get_weights()    

    def train(self,train_data,train_lable,epochs=10):
        self.model.fit(train_data,train_lable,epochs=epochs) 
        self.update_weights()
    
    def fisher_matrix(self,tasks):
        self.fisher = [tf.zeros(v.get_shape().as_list()) for v in self.model.weights]
        count = 0
        for data in tasks:
            count += len(data)//20
            for i in range(len(data)//20):
                d = tf.reshape(data[i],(1,28,28))
                with tf.GradientTape() as tape:
                    tape.watch(self.model.weights)
                    probs = self.model(d)
                    y = tf.math.log(probs)
                grad = tape.gradient(y,[v for v in self.model.weights])
                for v in range(len(self.fisher)):
                    self.fisher[v] += tf.square(grad[v])

        for v in range(len(self.fisher)):
            self.fisher[v] /= count
        return self.fisher    
    
    def custom_loss(self,y_true,y_pred):
        loss = self.loss_fn(y_true,y_pred)
        for v in range(len(self.theta)):
            loss += (self.lambda_val/2)*tf.reduce_sum(tf.multiply(self.fisher[v],tf.square(self.theta[v]-self.theta_star[v])))
        return loss        

    def train_with_ewc(self,train_data,train_label,validation_data,validation_label,epochs=10):
        self.fisher = self.fisher_matrix(validation_data) 

        self.model.compile(loss=self.custom_loss,optimizer='adam',metrics=['accuracy'])
        cb = CustomCallback()
        self.model.fit(train_data,train_label,epochs=epochs,validation_data=(validation_data[-1],validation_label[-1]),callbacks=[cb])
        if len(validation_data) == 2:
            acc_C = self.model.evaluate(train_data,train_label)[1]
            acc_B = self.model.evaluate(validation_data[1],validation_label[1])[1]
            acc_A = self.model.evaluate(validation_data[0],validation_label[0])[1]

            print('\nTask A:',acc_A)
            print('Task B:',acc_B)
            print('Task C:',acc_C)

        self.update_weights()    

if __name__ == '__main__':
    
    taskA,taskB,taskC = Dataset().get_data()
    nn = Continual_Learning_Neural_net(lambda_val=5)

    print("\nLearning Task A\n")
    nn.train(taskA['data']/255.0,taskA['label'])

    print("\nLearning Task B without forgetting Task A\n")
    nn.train_with_ewc(taskB['data']/255.0,taskB['label'],[taskA['data']/255.0],[taskA['label']],10)

    print("\nLearning Task C without forgetting Task A and Task B\n")
    nn.train_with_ewc(taskC['data']/255.0,taskC['label'],[taskA['data']/255.0,taskB['data']/255.0],[taskA['label'],taskB['label']],10)