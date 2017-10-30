import pandas as pd
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


class Processor:

	def __init__(self):
		self.df = None

	def read_data(self, csv_file):
		self.df = pd.read_csv(csv_file)

	def clean_prepare_data(self):

		clinton_error_data = []
		trump_error_data = []

		adj_clinton = self.df['adjpoll_clinton'].tolist()
		raw_clinton = self.df['rawpoll_clinton'].tolist()

		adj_trump = self.df['adjpoll_trump'].tolist()
		raw_trump = self.df['rawpoll_trump'].tolist()

		'''
		 Going through both columns of adj_clinton, raw_clinton
		 and storing the error in actual and trump_adjusted
		'''

		for k, v in zip(adj_clinton, raw_clinton):
			clinton_error_data.append(k - v)

		for k, v in zip(adj_trump, raw_trump):
			trump_error_data.append(k - v)

		'''
		fig=plt.figure()
		plt.plot(raw_clintion,clinton_error_data,color='black')
		plt.xlabel('This is the x axis')
		plt.ylabel('This is the y axis')
		plt.show()
		'''

		'''
		colors = np.random.rand(len(clinton_error_data))
		plt.scatter(raw_clintion, clinton_error_data,  c=colors)
		plt.xlabel('raw_clintion')
		plt.ylabel('clinton_error_data')
		plt.show()
		'''

		# finding out the number of days between startdate and today's date
		enddate_list = self.df['startdate'].tolist()
		days_from_poll = []

		TODAY = datetime.date(year=2016, month=12, day=12)
		for v in enddate_list:
			dates = datetime.datetime.strptime(v, '%m/%d/20%y').date()
			days_from_poll.append((TODAY - dates).days)

		'''
		colors = np.random.rand(len(clinton_error_data))
		plt.scatter(days_from_poll, clinton_error_data,  c=colors)
		plt.xlabel('days_from_poll')
		plt.ylabel('clinton_error_data')
		plt.show()
		'''

		binomial_data = []
		clinton_win = 1
		trump_win = 0

		"""
		Comparing adj_clinton and adj_trump, if it is 
		greater than 0 then Clinton wins
		and if adj_clinton and adj_trump is less than 0 then Trump wins
		"""
		for k, v in zip(adj_clinton, adj_trump):
			if k - v >= 0:
				binomial_data.append(clinton_win)
			else:
				binomial_data.append(trump_win)

		# Creating weighted_data list
		weighted_data = []

		'''
			colors = np.random.rand(2)
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')

			ax.scatter(clinton_error_data, binomial_data, days_from_poll, c= 'r', marker='o')

			ax.set_xlabel('clinton_error_data')
			ax.set_ylabel('winner')
			ax.set_zlabel('days from poll')
			plt.show()

		'''
		'''
		 set is used to remove all duplicate state names
		 now we assign each state name present here a number
		 eg :  {'Wisconsin': 43, 'Mississippi': 0, 'Washington': 49 }
		'''
		index = 0
		states_map = {}
		for st in set(self.df.state):
			states_map[st] = index
			index += 1
		state_data = []

		"""
		Creating state table with the map
		"""

		for st in self.df.state:
			state_data.append(states_map[st])

		"""
		 Creating a weighted data with samplesize, days from poll ,
		 polling weight
		"""

		count = 0
		for weight in self.df.poll_wt:
			tmp_weight = weight
			tmp_weight *= self.df.samplesize[count]
			tmp_weight /= days_from_poll[count]
			weighted_data.append(tmp_weight)
			count += 1

		"""
		 Final data preparation , with 5 columns
		"""

		table = {'trump_clinton_win': binomial_data, 'state': state_data,
		         'polling_weight': weighted_data,
		         'trump_adjusted': trump_error_data,
		         'clinton_adjusted': clinton_error_data}
		new_df = pd.DataFrame(data=table)

		return new_df

	def draw_visualization(self):
		dates = []
		for v in self.df.createddate:
			date_tmp = datetime.datetime.strptime(v, '%m/%d/20%y').date()
			dates.append(date_tmp)

		# adjustment data of Clinton and Trump
		trump_adj = self.df.adjpoll_trump
		clinton_adj = self.df.adjpoll_clinton

		fig, ax = plt.subplots(1, 1)
		'''
		 for scatter plot use ax.scatter
		 for bar plot use ax.bar
		'''
		ax.scatter(dates, clinton_adj, color='b')
		ax.xaxis.set_major_locator(mdates.MonthLocator())
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.20%y'))
		ax.set_xlabel('Poll end date')
		ax.set_ylabel('clinton adjusted')
		plt.show()
		fig, ax = plt.subplots(1, 1)
		ax.scatter(dates, trump_adj, color='r')
		ax.xaxis.set_major_locator(mdates.MonthLocator())
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.20%y'))
		ax.set_xlabel('Poll end date')
		ax.set_ylabel('trump adjusted')
		plt.show()

		'''
		fig, ax1 = plt.subplots()

		ax2 = ax1.twinx()
		ax2.bar(dates, trump_adj, color='r')
		ax2.xaxis.set_major_locator(mdates.MonthLocator())
		ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.20%y'))
		ax1.bar(dates, clinton_adj, color='g')
		ax1.xaxis.set_major_locator(mdates.MonthLocator())
		ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.20%y'))

		ax1.set_ylabel('clinton adjusted data')
		ax2.set_ylabel('trump adjusted data')
		plt.show()
		'''
