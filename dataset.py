import tensorflow as tf

class Dataset:
	def __init__(self):
		(self.train_data,self.train_label),(self.test_data,self.test_label) = tf.keras.datasets.mnist.load_data()
		self.train_label_A = tf.squeeze(tf.one_hot(self.train_label[self.train_label==0],depth=10))
		self.train_label_B = tf.squeeze(tf.one_hot(self.train_label[self.train_label==1],depth=10))
		self.train_label_C = tf.squeeze(tf.one_hot(self.train_label[self.train_label==2],depth=10))

	def get_data(self):

		taskA = {'data':self.train_data[self.train_label==0],'label':self.train_label_A}
		taskB = {'data':self.train_data[self.train_label==1],'label':self.train_label_B}
		taskC = {'data':self.train_data[self.train_label==2],'label':self.train_label_C}
		return taskA,taskB,taskC

if __name__ == '__main__':
	d = Dataset()
	tA,tB,tC = d.get_data()