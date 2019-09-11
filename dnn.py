# Author: Ljubisa Sehovac
# github: sehovaclj

# code that uses a DNN to forecast energy consumption
# refer to papers for more details


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

from tempfile import TemporaryFile


#  building the model

class NNModel(nn.Module):
	def __init__(self, input_size, output_size, deep_size):
		super(NNModel, self).__init__()

		self.input_size = input_size
		self.output_size = output_size
		self.deep_size = deep_size

		if self.deep_size == 'small':
			self.input_layer = nn.Linear(self.input_size, 512)
			self.layer1 = nn.Linear(512, 256)
			self.layer2 = nn.Linear(256, 128)
			self.output_layer = nn.Linear(128, self.output_size)

		if self.deep_size == 'medium':
			self.input_layer = nn.Linear(self.input_size, 512)
			self.layer1 = nn.Linear(512, 512)
			self.layer2 = nn.Linear(512, 512)
			self.layer3 = nn.Linear(512, 256)
			self.layer4 = nn.Linear(256, 128)
			self.output_layer = nn.Linear(128, self.output_size)

		if self.deep_size == 'large':
			self.input_layer = nn.Linear(self.input_size, 1024)
			self.layer1 = nn.Linear(1024, 1024)
			self.layer2 = nn.Linear(1024, 512)
			self.layer3 = nn.Linear(512, 512)
			self.layer4 = nn.Linear(512, 512)
			self.layer5 = nn.Linear(512, 256)
			self.layer6 = nn.Linear(256, 256)
			self.output_layer = nn.Linear(256, self.output_size)


		self.init()


	# how important is this init function? do with and without
	# function to intialize weight parameters
	def init(self):
		for p in self.parameters():
			if p.dim() > 1:
				init.orthogonal_(p.data, gain=1.0)

			if p.dim() == 0:
				init.constant(p.data, 0.0)


	def forward(self, x):

		if self.deep_size == 'small':
			out = self.output_layer(self.layer2(self.layer1(self.input_layer(x))))
			return out

		if self.deep_size == 'medium':
			out = self.output_layer(self.layer4(self.layer3(self.layer2(self.layer1(self.input_layer(x))))))
			return out

		if self.deep_size == 'large':
			out = self.output_layer(self.layer6(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(self.input_layer(x))))))))
			return out


#################################################################################################################################################

# main function

def main(seed, cuda, deep_size, window_source_size, window_target_size, epochs, batch_size):

	t0 = time.time()

	# seed=1
	np.random.seed(seed)
	torch.manual_seed(seed)

	print("Loading dataset")
	d = np.loadtxt("./Anonymous_dataset.csv", delimiter=",", skiprows=1, dtype=str)
	
	dataset = d[:, 4:].astype(np.float32)

	dataset = pd.DataFrame(dataset)
	dataset.columns = ['month', 'day_of_year', 'day_of_month', 'weekday', 'weekend', 'holiday', 'hour', 'minute', 'season', 'temp', 'hum', 'usage']

	dataset = dataset.drop('minute',1)

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
	print("Generating training and test data...")
	WINDOW_SOURCE_SIZE = window_source_size
	WINDOW_TARGET_SIZE = window_target_size

	# actual usage
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
	INPUT_SIZE = WINDOW_SOURCE_SIZE*X_train.shape[-1]
	OUTPUT_SIZE = Y_train_usage.shape[1]
	DEEP_SIZE = deep_size

	model = NNModel(INPUT_SIZE, OUTPUT_SIZE, DEEP_SIZE)

	if cuda:
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		model.cuda()


	print("MODEL ARCHITECTURE IS: ")
	print(model)

	opt = optim.Adam(model.parameters(), lr=1e-3)
	loss_fn = nn.MSELoss(reduction='sum')
	EPOCHES = epochs
	BATCH_SIZE = batch_size

	train_loss = []
	test_loss = []

	print("\nStarting training...")

	for epoch in range(EPOCHES):

		t_one_epoch = time.time()

		print("Epoch {}".format(epoch+1))

		total_usage_loss = 0

		for b_idx in range(0, X_train.shape[0], BATCH_SIZE):

			x = X_train[b_idx:b_idx+BATCH_SIZE]
			y_usage = Y_train_usage[b_idx:b_idx+BATCH_SIZE]

			x = np.reshape(x, (x.shape[0], INPUT_SIZE))
			y_usage = np.reshape(y_usage, (y_usage.shape[0], OUTPUT_SIZE))

			# convert to torch
			x = torch.from_numpy(x).float()
			y_usage = torch.from_numpy(y_usage).float()

			if cuda:
				x = x.cuda()
				y_usage = y_usage.cuda()

			# pass through Deep NN
			out = model.forward(x)

			loss_usage = loss_fn(out, y_usage)

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

		total_usage_loss = 0

		all_preds = []

		for b_idx in range(0, X_test.shape[0], BATCH_SIZE):
			with torch.no_grad():

				x = X_test[b_idx:b_idx+BATCH_SIZE]
				y_usage = Y_test_usage[b_idx:b_idx+BATCH_SIZE]

				x = np.reshape(x, (x.shape[0], INPUT_SIZE))
				y_usage = np.reshape(y_usage, (y_usage.shape[0], OUTPUT_SIZE))

				# convert to torch
				x = torch.from_numpy(x).float()
				y_usage = torch.from_numpy(y_usage).float()

				if cuda:
					x = x.cuda()
					y_usage = y_usage.cuda()

				# pass through Deep NN
				out = model.forward(x)

				loss_usage = loss_fn(out, y_usage)

				total_usage_loss += loss_usage.item()

				if (epoch == epochs-1):
					all_preds.append(out)

		test_loss.append(total_usage_loss)

		print("\tTESTING: {} total test USAGE loss".format(total_usage_loss))

		print("\tTESTING:\n")
		print("\tSample of prediction:")
		print("\t\t TARGET: {}".format(y_usage[-1].cpu().detach().numpy().flatten()))
		print("\t\t   PRED: {}\n\n".format(out[-1].cpu().detach().numpy().flatten()))

		y_last_usage = y_usage[-1].cpu().detach().numpy().flatten()
		pred_last_usage = out[-1].cpu().detach().numpy().flatten()

		t2_one_epoch = time.time()

		time_one_epoch = t2_one_epoch - t_one_epoch

		print("TIME OF ONE EPOCH: {} seconds and {} minutes".format(time_one_epoch, time_one_epoch/60.0))




	#################################################################################################################################################
	# PLOTTING

	# for plotting and accuracy
	preds = torch.cat(all_preds, 0)
	preds = preds.cpu().detach().numpy().flatten()

	actual = Y_test_usage.flatten()

	train_loss_array = np.asarray(train_loss)
	test_loss_array = np.asarray(test_loss)

	len_loss = np.arange(len(train_loss_array))

	preds_unnorm = (preds*std_usage) + mu_usage

	# accuracy measure MAE and MAPE
	mae2 = (sum(abs(usage_actual - preds_unnorm)))/(len(usage_actual))
	mape2 = (sum(abs((usage_actual - preds_unnorm)/usage_actual)))/(len(usage_actual))

	# for std
	mae_s = abs(usage_actual - preds_unnorm)
	s2 = mae_s.std()

	mape_s = abs((usage_actual - preds_unnorm)/usage_actual)
	s = mape_s.std()



	print("\tMAE, and MAPE: {} and {}%\n".format(mae2, mape2*100.0))



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

	for_plotting = [usage_actual, preds_unnorm, y_last_usage, pred_last_usage]


	return s, s2, mape_s, mae_s, mae2, mape2, total/60.0, train_loss, test_loss, for_plotting










###################################################################################################################

# call main function

if __name__ == "__main__":

    s, s2, mape_s, mae_s, mae, mape, total_mins, train_loss, test_loss, fp = main(seed=0, cuda=True,
		deep_size='small', window_source_size=12, window_target_size=6,
		epochs=1, batch_size=256)
























