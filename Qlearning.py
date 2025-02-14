
import random
import numpy as np

class RandomPlay():
	def __init__(self, player_num):
		self.player_num = player_num

	def get_move(self, environ):

		legal_mode_mask = GetLegalMoveMask(environ)
		q_rand = np.random.rand(9) * 2 - 1
		q_rand[legal_mode_mask == 0] = np.nan

		action = np.nanargmax(q_rand)
		return action, action // 3, action % 3

class QLearnPlay():
	def __init__(self, player_num, fina=None, fina_keys=None, 
			swap_player_ids = False, alpha = 0.7, gamma = 0.95,
			epsilon = 0.05):

		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon #random move probability

		self.player_num = player_num
		self.q = []
		self.q_state_keys = []
		self.q_state_dict = {}

		if fina is not None:
			with open(fina, 'rb') as f:
				self.q = np.load(f)
		if fina_keys is not None:
			with open(fina_keys, 'rb') as f:
				self.q_state_keys = np.load(f)

		# Depending on which payer id was used in traning, the ids might need to be swapped
		if self.q_state_keys is not None and swap_player_ids:
			self.q_state_keys[self.q_state_keys > 0] = 3 - self.q_state_keys[self.q_state_keys > 0]

		self.q_state_dict = {}
		for k, q_row in zip(self.q_state_keys, self.q):
			self.q_state_dict[tuple(k)] = q_row

	def set_epsilon(self, val):
		self.epsilon = val

	def get_move(self, environ):

		s = tuple(environ.flatten())
		legal_mode_mask = GetLegalMoveMask(environ)

		# Choose an action (Epsilon-greedy policy)
		if random.random() < self.epsilon:
			# Exploration with random action
			legal_moves = GetLegalMoveMask(environ)
			q_rand = np.random.rand(9) * 2 - 1
			q_rand[legal_mode_mask == 0] = np.nan

			action = np.nanargmax(q_rand)
			return action, action // 3, action % 3

		# Planned move (exploitation)
		if s in self.q_state_dict:

			#Found matching state
			q_row = self.q_state_dict[s].copy()
			q_row[legal_mode_mask == 0] = np.nan
			action = np.nanargmax(q_row)
			return action, action // 3, action % 3

		else:

			# Look for a similar state
			min_diff = None
			best_rows = []
			for row_state, q_row in zip(self.q_state_keys, self.q):
				row_state = row_state.flatten()
				row_diff = np.sum(np.abs(s - row_state))

				if min_diff is None or row_diff < min_diff:
					min_diff = row_diff
					best_rows = [row_state]
				elif row_diff == min_diff:
					best_rows.append(row_state)

			best_row = random.choice(best_rows)

			q_row = self.q_state_dict[tuple(best_row)].copy()
			q_row[legal_mode_mask == 0] = np.nan
			action = np.nanargmax(q_row)
			return action, action // 3, action % 3

	def train(self, old_environ, action_id, environ, reward):

		old_q_row = self.q_state_dict[tuple(old_environ.flatten())]

		self.check_state_exists(environ)

		q_row = self.q_state_dict[tuple(environ.flatten())]

		# Get max reward of available states
		max_future_reward = np.max(q_row)

		# Update q table using the Bellman equation
		new_q = old_q_row[action_id] + self.alpha * (reward + self.gamma * max_future_reward - old_q_row[action_id])
		old_q_row[action_id] = new_q

	def check_state_exists(self, environ):

		# Create state in q table
		environ_flat = tuple(environ.reshape((9,)))
		if environ_flat not in self.q_state_dict:
			q_row = np.random.rand(9) * 2 - 1
			
			self.q_state_keys.append(environ_flat)
			self.q_state_dict[environ_flat] = q_row
			self.q.append(q_row)
		else:
			q_row = self.q_state_dict[environ_flat]

	def save(self):
		with open('q_table.npy', 'wb') as f:
			np.save(f, np.array(self.q))
		with open('q_table_keys.npy', 'wb') as f:
			np.save(f, np.array(self.q_state_keys))


def CheckForWin(environ):
	for r in range(environ.shape[0]):

		if environ[r, 0] != 0:

			win = True
			for c in range(1, environ.shape[0]):
				if environ[r, c] != environ[r, 0]:
					win = False
					break
			if win:
				return environ[r, 0]

	for c in range(environ.shape[0]):

		if environ[0, c] != 0:

			win = True
			for r in range(1, environ.shape[1]):
				if environ[r, c] != environ[0, c]:
					win = False
					break
			if win:
				return environ[0, c]

	if environ[0, 0] != 0:

		win = True
		for d in range(1, environ.shape[0]):
			if environ[d, d] != environ[0, 0]:
				win = False
				break
		if win:
			return environ[0, 0]

	if environ[0, -1] != 0:

		win = True
		for d in range(1, environ.shape[0]):
			if environ[d, environ.shape[1]-d-1] != environ[0, -1]:
				win = False
				break
		if win:
			return environ[0, -1]
	
	return 0

def GetLegalMoves(environ):
	m = np.where(environ == 0)
	return m

def GetLegalMoveMask(environ):
	m = environ.flatten() == 0
	return m

def run():

	max_epsilon = 1.0
	min_epsilon = 0.05
	max_steps = 100000

	train_player = QLearnPlay(1)
	random_train = True
	if random_train:
		opponent = RandomPlay(2)
	else:
		opponent = QLearnPlay(2, fina='q_table1.npy', fina_keys='q_table_keys1.npy', 
			swap_player_ids = True)

	for i in range(max_steps):

		epsilon = max_epsilon + (min_epsilon - max_epsilon) * i / max_steps
		environ = np.zeros((3,3), dtype=np.int8)
		train_player.set_epsilon(epsilon)
		if not random_train:
			opponent.set_epsilon(epsilon)

		if random.randint(0,1) != 0:
			# Opponent goes first
			action2_id, move2_x, move2_y = opponent.get_move(environ)
			environ[move2_x, move2_y] = opponent.player_num

		train_player.check_state_exists(environ)

		while True:

			old_environ = environ.copy()

			action_id, move1_x, move1_y = train_player.get_move(environ)

			# Perform action

			assert environ[move1_x, move1_y] == 0
			environ[move1_x, move1_y] = train_player.player_num

			win = CheckForWin(environ)
			if win == 1:
				print (i, "player 1 wins")
			else:

				legal_moves_x, legal_moves_y = GetLegalMoves(environ)
				if len(legal_moves_x) == 0: # draw
					print (i, "draw")
					win = -1
				else:
					action2_id, move2_x, move2_y = opponent.get_move(environ)
					assert environ[move2_x, move2_y] == 0	
					environ[move2_x, move2_y] = 2

					win = CheckForWin(environ)
					if win == opponent.player_num:
						print (i, "player 2 wins")

					else:
						legal_moves_x, legal_moves_y = GetLegalMoves(environ)
						if len(legal_moves_x) == 0: # draw
							print (i, "draw")
							win = -1

			# Measure reward
			rewardDict = {-1:0, 0:0, 1:1, 2:-1}
			reward = rewardDict[win]

			train_player.train(old_environ, action_id, environ, reward)

			if win != 0:
				break #Game over

		print ("q table len", len(train_player.q), ", epsilon", epsilon)

	train_player.save()

if __name__=="__main__":
	run()

