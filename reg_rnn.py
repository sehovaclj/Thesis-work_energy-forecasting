# Author: Ljubisa Sehovac
# github: sehovaclj

# code that uses a regular RNN to forecast energy consumption. Refer to Journal paper for more details

# importing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init

import random
import time

import math

#from twilio.rest import Client

from tempfile import TemporaryFile




#########################################################################################

# building the S2S model

class SModel(nn.Module):
	def __init__(self, cell_type, input_size, hidden_size, use_cuda, pred_type, pred_length):
		super(SModel, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size

		self.cell_type = cell_type

		if self.cell_type not in ['rnn', 'gru', 'lstm']:
			raise ValueError(self.cell_type, " is not an appropriate cell type. Please select one of rnn, gru, or lstm.")

		if self.cell_type == 'rnn':
			self.rnncell = nn.RNNCell(self.input_size, self.hidden_size)

		if self.cell_type == 'gru':
			self.rnncell = nn.GRUCell(self.input_size, self.hidden_size)

		if self.cell_type == 'lstm':
			self.rnncell = nn.LSTMCell(self.input_size, self.hidden_size)


		self.lin_usage = nn.Linear(self.hidden_size, 1)

		self.use_cuda = use_cuda

		self.pred_length = pred_length
		self.pred_type = pred_type

		self.init()


	# VERY IMPORTANT INIT PARAMS FUNCTIONS***
	# function to intialize weight parameters

	def init(self):

		if self.cell_type == 'rnn' or self.cell_type == 'gru':
			#j = []
			for p in self.parameters():
				if p.dim() > 1:
					init.orthogonal_(p.data, gain=1.0)
					#j.append(p.data)
				if p.dim() == 1:
					init.constant_(p.data, 0.0)
					#j.append(p.data)

		elif self.cell_type == 'lstm':
			#j = []
			for p in self.parameters():
				if p.dim() > 1:
					init.orthogonal_(p.data, gain=1.0)
					#j.append(p.data)
				if p.dim() == 1:
					init.constant_(p.data, 0.0)
					init.constant_(p.data[self.hidden_size:2*self.hidden_size], 1.0)
					#j.append(p.data)
		#return j



	def forward(self, x, pred_type, pred_length):
		# encoder forward function

		self.pred_type = pred_type
		self.pred_length = pred_length

		preds = []

		# for rnn and gru
		if self.cell_type == 'rnn' or self.cell_type == 'gru':

			h = torch.zeros(x.shape[0], self.hidden_size)

			if self.use_cuda:
				h = h.cuda()

			if self.pred_type == 'full':

				for T in range(x.shape[1]):
					h = self.rnncell(x[:, T, :], h)
					pred_usage = self.lin_usage(h)
					preds.append(pred_usage.unsqueeze(1))

				preds = torch.cat(preds, 1)

			elif self.pred_type == 'partial':

				for T in range(x.shape[1]):
					h = self.rnncell(x[:, T, :], h)
					if T >= (x.shape[1] - self.pred_length):
						pred_usage = self.lin_usage(h)
						preds.append(pred_usage.unsqueeze(1))

				preds = torch.cat(preds, 1)




		# for lstm
		elif self.cell_type == 'lstm':

			h0 = torch.zeros(x.shape[0], self.hidden_size)
			c0 = torch.zeros(x.shape[0], self.hidden_size)

			if self.use_cuda:
				h0 = h0.cuda()
				c0 = c0.cuda()

			h = (h0, c0)

			if self.pred_type == 'full':

				for T in range(x.shape[1]):
					h = self.rnncell(x[:, T, :], h)
					pred_usage = self.lin_usage(h[0])
					preds.append(pred_usage.unsqueeze(1))

				preds = torch.cat(preds, 1)

			elif self.pred_type == 'partial':

				for T in range(x.shape[1]):
					h = self.rnncell(x[:, T, :], h)
					if T >= (x.shape[1] - self.pred_length):
						pred_usage = self.lin_usage(h[0])
						preds.append(pred_usage.unsqueeze(1))

				preds = torch.cat(preds, 1)



		return preds






#################################################################################################################################################

# main function

def main(seed, cuda, cell_type, window_source_size, window_target_size, pred_type, pred_length, epochs, batch_size, hs):

	t0 = time.time()

	# seed == given seed
	np.random.seed(seed)
	torch.manual_seed(seed)

	print("Loading dataset...")
	d = np.loadtxt("./Anonymous_dataset.csv", delimiter=",", skiprows=1, dtype=str)

	dataset = d[:, 4:].astype(np.float32)

	dataset = pd.DataFrame(dataset)
	dataset.columns = ['month', 'day_of_year', 'day_of_month', 'weekday', 'weekend',
                       'holiday', 'hour', 'minute', 'season', 'temp', 'hum', 'usage']

	# switch around columns
	dataset = dataset[['month', 'day_of_year','day_of_month', 'season',
             'weekday', 'weekend', 'holiday', 'hour', 'minute', 'temp', 'hum', 'usage']]

	dataset = dataset.drop('minute',1).drop('temp',1).drop('hum',1)

	usage_actual = dataset['usage']

	mu_usage = dataset['usage'].mean()
	std_usage = dataset['usage'].std()

	dataset = dataset.values


	# 0 mean and unit var
	print("Transforming data to 0 mean and unit var")
	MU = dataset.mean(0) # 0 means take the mean of the column
	dataset = dataset - MU

	STD = dataset.std(0) # same with std here
	dataset = dataset / STD

	# 5 minutes between rows.
	# use 1 hour (12 rows) to predict next half hour (6 rows)
	#print("Generating training and test data...")
	WINDOW_SOURCE_SIZE = window_source_size
	WINDOW_TARGET_SIZE = window_target_size


	# getting actual usage vector, aligning with predicted values vector. Aka remove first window_source_size and remaining
	usage_actual = usage_actual.values
	usage_actual = usage_actual[int(dataset.shape[0]*0.80):]
	usage_actual = usage_actual[WINDOW_SOURCE_SIZE:]

	# 80% of the data will be train
	# 20% is test
	train_source = dataset[:int(dataset.shape[0]*0.80)]
	test_source = dataset[int(dataset.shape[0]*0.80):]

	# if N = data.shape[0] - (WINDOW_SOURCE_SIZE+WINDOW_TARGET_SIZE)
	# then you will be sliding over every one

	def generate_windows(data):
		x_train = []
		y_usage_train = []

		x_test = []
		y_usage_test = []

		# for training data
		idxs = np.random.choice(train_source.shape[0]-(WINDOW_SOURCE_SIZE+WINDOW_TARGET_SIZE), train_source.shape[0]-(WINDOW_SOURCE_SIZE+WINDOW_TARGET_SIZE), replace=False)

		for idx in idxs:
			x_train.append(train_source[idx:idx+WINDOW_SOURCE_SIZE].reshape((1, WINDOW_SOURCE_SIZE, train_source.shape[1])) )
			y_usage_train.append(train_source[idx+WINDOW_SOURCE_SIZE:idx+WINDOW_SOURCE_SIZE+WINDOW_TARGET_SIZE, -1].reshape((1, WINDOW_TARGET_SIZE, 1)) )

		x_train = np.concatenate(x_train, axis=0) # make them arrays and not lists
		y_usage_train = np.concatenate(y_usage_train, axis=0)

		# for testing data
		idxs = np.arange(0, len(test_source)-(WINDOW_SOURCE_SIZE+WINDOW_TARGET_SIZE), WINDOW_TARGET_SIZE)

		for idx in idxs:
			x_test.append(test_source[idx:idx+WINDOW_SOURCE_SIZE].reshape((1, WINDOW_SOURCE_SIZE, test_source.shape[1])) )
			y_usage_test.append(test_source[idx+WINDOW_SOURCE_SIZE:idx+WINDOW_SOURCE_SIZE+WINDOW_TARGET_SIZE, -1].reshape((1, WINDOW_TARGET_SIZE, 1)) )

		x_test = np.concatenate(x_test, axis=0) # make them arrays and not lists
		y_usage_test = np.concatenate(y_usage_test, axis=0)

		return x_train, y_usage_train, x_test, y_usage_test


	X_train, Y_train_usage, X_test, Y_test_usage = generate_windows(dataset)
	print("Created {} train samples and {} test samples".format(X_train.shape[0], X_test.shape[0]))


	idxs = np.arange(0, len(test_source)-(WINDOW_SOURCE_SIZE+WINDOW_TARGET_SIZE), WINDOW_TARGET_SIZE)
	remainder = len(test_source) - (idxs[-1] + WINDOW_SOURCE_SIZE+WINDOW_TARGET_SIZE)

	usage_actual = usage_actual[:-remainder]




	#################################################################################################################################################

	# call the model

	print("Creating model...")
	INPUT_SIZE = X_train.shape[-1]
	HIDDEN_SIZE = hs
	CELL_TYPE = cell_type
	PRED_TYPE = pred_type
	PRED_LENGTH = pred_length


	model = SModel(CELL_TYPE, INPUT_SIZE, HIDDEN_SIZE, cuda, PRED_TYPE, PRED_LENGTH)


	if cuda:
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		model.cuda()


	print("MODEL ARCHITECTURE IS: ")
	print(model)

	print("\nModel parameters are on cuda: {}".format(next(model.parameters()).is_cuda))

	opt = optim.Adam(model.parameters(), lr=1e-3)
	loss_fn = nn.MSELoss(reduction='sum')
	EPOCHES = epochs
	BATCH_SIZE = batch_size


	print("\nStarting training...")

	train_loss = []
	test_loss = []

	all_mae = []
	all_mape = []


	for epoch in range(EPOCHES):

		t_one_epoch = time.time()

		print("Epoch {}".format(epoch+1))

		total_usage_loss = 0

		for b_idx in range(0, X_train.shape[0], BATCH_SIZE):

			x = torch.from_numpy(X_train[b_idx:b_idx+BATCH_SIZE]).float()
			y_usage = torch.from_numpy(Y_train_usage[b_idx:b_idx+BATCH_SIZE]).float()

			if cuda:
				x = x.cuda()
				y_usage = y_usage.cuda()


			preds = model.forward(x, PRED_TYPE, PRED_LENGTH)

			# compute lose
			loss_usage = loss_fn(preds, y_usage)

			# backprop and update
			opt.zero_grad()

			loss_usage.backward()

			opt.step()

			total_usage_loss += loss_usage.item()


		train_loss.append(total_usage_loss)

		print("\tTRAINING: {} total train USAGE loss.\n".format(total_usage_loss))


		#################################################################################################################################################
		# TESTING

		y_usage = None
		pred_usage = None
		preds = None

		total_usage_loss = 0

		all_preds = []

		all_preds2 = []


		for b_idx in range(0, X_test.shape[0], BATCH_SIZE):
			with torch.no_grad():

				x = torch.from_numpy(X_test[b_idx:b_idx+BATCH_SIZE])
				y_usage = torch.from_numpy(Y_test_usage[b_idx:b_idx+BATCH_SIZE])

				if cuda:
					x = x.cuda()
					y_usage = y_usage.cuda()

				preds = model.forward(x, PRED_TYPE, PRED_LENGTH)

				# compute loss
				loss_usage = loss_fn(preds, y_usage)

				total_usage_loss += loss_usage.item()

				if (epoch == epochs-1):
					all_preds.append(preds)


				all_preds2.append(preds)
		preds_formape = torch.cat(all_preds2, 0)
		preds_formape = preds_formape.cpu().detach().numpy().flatten()

		preds_unnorm_formape = (preds_formape*std_usage) + mu_usage

		mae1 = (sum(abs(usage_actual - preds_unnorm_formape)))/(len(usage_actual))
		mape1 = (sum(abs((usage_actual - preds_unnorm_formape)/usage_actual)))/(len(usage_actual))

		all_mae.append(mae1)
		all_mape.append(mape1)


		test_loss.append(total_usage_loss)

		print("\tTESTING: {} total test USAGE loss".format(total_usage_loss))

		print("\tTESTING:\n")
		print("\tSample of prediction:")
		print("\t\t TARGET: {}".format(y_usage[-1].cpu().detach().numpy().flatten()))
		print("\t\t   PRED: {}\n\n".format(preds[-1].cpu().detach().numpy().flatten()))

		y_last_usage = y_usage[-1].cpu().detach().numpy().flatten()
		pred_last_usage = preds[-1].cpu().detach().numpy().flatten()

		t2_one_epoch = time.time()

		time_one_epoch = t2_one_epoch - t_one_epoch

		print("TIME OF ONE EPOCH: {} seconds and {} minutes".format(time_one_epoch, time_one_epoch/60.0))


	####################################################################################################
	# PLOTTING

	# for plotting and accuracy
	preds = torch.cat(all_preds, 0)
	preds = preds.cpu().detach().numpy().flatten()

	actual = Y_test_usage.flatten()

	# for loss plotting
	train_loss_array = np.asarray(train_loss)
	test_loss_array = np.asarray(test_loss)

	len_loss = np.arange(len(train_loss_array))

	# unnormalizing 1
	preds_unnorm = (preds*std_usage) + mu_usage


	# using the actual actual usage lol not as above and using unnormalized usage
	mae3 = (sum(abs(usage_actual - preds_unnorm)))/(len(usage_actual))
	mape3 = (sum(abs((usage_actual - preds_unnorm)/usage_actual)))/(len(usage_actual))

	# for std
	mae_s = abs(usage_actual - preds_unnorm)
	s2 = mae_s.std()

	mape_s = abs((usage_actual - preds_unnorm)/usage_actual)
	s = mape_s.std()



	print("\n\tACTUAL ACC. RESULTS: MAE, MAPE: {} and {}%".format(mae3, mape3*100.0))



	# plotting
	plt.figure(1)
	plt.plot(np.arange(len(preds)), preds, 'b', label='Predicted')
	plt.plot(np.arange(len(actual)), actual, 'g', label='Actual')
	plt.title("Predicted vs Actual, {} timesteps to {} timesteps".format(window_source_size, window_target_size))
	plt.xlabel("Time in 5 minute increments")
	plt.ylabel("Usage (normalized)")
	plt.legend(loc='lower left')

	plt.figure(2)
	plt.plot(np.arange(len(actual)), actual, 'g', label='Actual')
	plt.plot(np.arange(len(preds)), preds, 'b', label='Predicted')
	plt.title("Predicted vs Actual, {} timesteps to {} timesteps".format(window_source_size, window_target_size))
	plt.xlabel("Time in 5 minute increments")
	plt.ylabel("Usage (normalized)")
	plt.legend(loc='lower left')

	plt.figure(3)
	plt.plot(np.arange(len(y_last_usage)), y_last_usage, 'g', label='Actual')
	plt.plot(np.arange(len(pred_last_usage)), pred_last_usage, 'b', label='Predicted')
	plt.title("Predicted vs Actual last test example, {} timesteps to {} timesteps".format(window_source_size, window_target_size))
	plt.xlabel("Time in 5 minute increments")
	plt.ylabel("Usage (normalized)")
	plt.legend(loc='lower left')

	plt.figure(4)
	plt.plot(np.arange(len(usage_actual[-12*24*7:])), usage_actual[-12*24*7:], 'g', label='Actual')
	plt.plot(np.arange(len(preds_unnorm[-12*24*7:])), preds_unnorm[-12*24*7:], 'b', label='Predicted')
	plt.title("Predicted vs Actual: Case 2, Zoom last 7 days".format(window_source_size, window_target_size))
	plt.xlabel("Time in 5 minute increments")
	plt.ylabel("Usage (kW)")
	plt.legend(loc='lower left')

	plt.figure(5)
	plt.plot(len_loss, train_loss_array, 'k')
	plt.title("Train loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")

	plt.figure(6)
	plt.plot(len_loss, test_loss_array, 'r')
	plt.title("Test Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")



	# total time of run
	t1 = time.time()
	total = t1-t0
	print("\nTIME ELAPSED: {} seconds OR {} minutes".format(total, total/60.0))


	print("\nEnd of run")


	plt.show()


	


	return s, s2, mape_s, mae_s, mae3, mape3, total/60.0, train_loss, test_loss










#######################################################################################################################

# calling the main function


if __name__ == "__main__":

	s, s2, mape_s, mae_s, mae, mape, total_mins, train_loss, test_loss = main(seed=0, cuda=True,
		cell_type='gru', window_source_size=12, window_target_size=6,
		pred_type='partial', pred_length=6, epochs=1, batch_size=256, hs=64)














