import numpy as np
cimport numpy as np
import random
from libc.stdlib cimport rand, srand, RAND_MAX
import pandas as pd
from pandas import ExcelWriter
#rand_seed = np.random.randint(0, 100)
########################################
### The Regression Tsetlin Machine #####
########################################
cdef class TsetlinMachine:
	cdef int number_of_clauses
	cdef int number_of_features

	cdef float s
	cdef float max_target
	cdef float min_target
	cdef int number_of_states
	cdef int threshold

	cdef int[:,:,:] ta_state
	cdef int[:,:,:] saved_ta_state

	cdef int[:] clause_sign
	cdef int[:] saved_clause_sign

	cdef int[:] clause_output
	cdef int[:] saved_clause_output

	cdef int[:] feedback_to_clauses
	cdef int[:] saved_feedback_to_clauses

	# Initialization of the Regression Tsetlin Machine
	def __init__(self, number_of_clauses, number_of_features, number_of_states, s, threshold, max_target, min_target):
		cdef int j
		srand(42)
		self.number_of_clauses = number_of_clauses
		self.number_of_features = number_of_features
		self.number_of_states = number_of_states
		self.s = s
		self.threshold = threshold
		self.max_target = max_target
		self.min_target = min_target

		# The state of each Tsetlin Automaton is stored here. The automata are randomly initialized to either 'number_of_states' or 'number_of_states' + 1.
		self.ta_state = np.random.choice([self.number_of_states, self.number_of_states+1], size=(self.number_of_clauses, self.number_of_features, 2)).astype(dtype=np.int32)
		self.saved_ta_state = np.random.choice([self.number_of_states, self.number_of_states+1], size=(self.number_of_clauses, self.number_of_features, 2)).astype(dtype=np.int32)
		# Data structure for keeping track of the sign of each clause
		self.clause_sign = np.zeros(self.number_of_clauses, dtype=np.int32)
		self.saved_clause_sign = np.zeros(self.number_of_clauses, dtype=np.int32)
		# Data structures for intermediate calculations (clause output, summation of votes, and feedback to clauses)
		self.clause_output = np.zeros(shape=(self.number_of_clauses), dtype=np.int32)
		self.saved_clause_output = np.zeros(shape=(self.number_of_clauses), dtype=np.int32)

		self.feedback_to_clauses = np.zeros(shape=(self.number_of_clauses), dtype=np.int32)
		self.saved_feedback_to_clauses = np.zeros(shape=(self.number_of_clauses), dtype=np.int32)

		# Set up the Regression Tsetlin Machine structure
		for j in xrange(self.number_of_clauses):
			if j % 2 == 0:
				self.clause_sign[j] = 1
			else:
				self.clause_sign[j] = 1


	# Calculate the output of each clause using the actions of each Tsetline Automaton.
	# Output is stored an internal output array.
	cdef void calculate_clause_output(self, int[:] X):
		cdef int j, k

		for j in xrange(self.number_of_clauses):
			self.clause_output[j] = 1
			for k in xrange(self.number_of_features):
				action_include = self.action(self.ta_state[j,k,0])
				action_include_negated = self.action(self.ta_state[j,k,1])

				if (action_include == 1 and X[k] == 0) or (action_include_negated == 1 and X[k] == 1):
					self.clause_output[j] = 0
					break

	###########################################
	### Predict Target Output y for Input X ###
	###########################################

	cpdef float predict(self, int[:] X):
		cdef float output_sum
		cdef float output_value
		cdef int j
		#print(f'ta_state: {list(self.ta_state[0][0])}')
		#print(f'saved_ta_state: {list(self.saved_ta_state[0][0])}')
		#print('')
		#print(f'clause_sign: {list(self.clause_sign)}')
		#print(f'clause_output: {list(self.clause_output)}')
		#print(f'feedback_to_clauses: {list(self.feedback_to_clauses)}')

		###############################
		### Calculate Clause Output ###
		###############################

		self.calculate_clause_output(X)

		###########################
		### Sum up Clause Votes ###
		###########################

        # Map the total clause outputs into a continuous value using max and min values of the target series
		output_sum = self.sum_up_clause_votes()
		output_value = ((output_sum * (self.max_target-self.min_target))/ self.threshold) + self.min_target

		return output_value

	# Translates automata state to action
	cdef int action(self, int state):
		if state <= self.number_of_states:
			return 0
		else:
			return 1

	# Get the state of a specific automaton, indexed by clause, feature, and automaton type (include/include negated).
	def get_state(self, int clause, int feature, int automaton_type):
		return self.self.ta_state[clause,feature,automaton_type]

	# Sum up the votes for each output
	cdef int sum_up_clause_votes(self):
		cdef int output_sum
		cdef int j

		output_sum = 0
		for j in xrange(self.number_of_clauses):
			output_sum += self.clause_output[j]*self.clause_sign[j]

		if output_sum > self.threshold:
			output_sum = self.threshold

		elif output_sum < 0:
			output_sum = 0

		return output_sum

	#######################################################
	### Evaluate the Trained Regression Tsetlin Machine ###
	#######################################################

	def evaluate(self, int[:,:] X, float[:] y):
		cdef int j,l
		cdef float errors
		cdef int output_sum
		cdef int[:] Xi
		number_of_examples = len(y)
		Xi = np.zeros((self.number_of_features,), dtype=np.int32)

		errors = 0
		for l in xrange(number_of_examples):
			###############################
			### Calculate Clause Output ###
			###############################

			for j in xrange(self.number_of_features):
				Xi[j] = X[l,j]

			errors += abs(self.predict(Xi) - y[l])

		return errors / number_of_examples

	#####################################################
	### Online Training of Regression Tsetlin Machine ###
	#####################################################

	# The Regression Tsetlin Machine can be trained incrementally, one training example at a time.
	# Use this method directly for online and incremental training.

	cpdef void update(self, int[:] X, float y):
		cdef int i, j
		cdef int action_include, action_include_negated
		cdef float output_sum
		cdef float output_value
		feedback = -1
		###############################
		### Calculate Clause Output ###
		###############################

		self.calculate_clause_output(X)

		###########################
		### Sum up Clause Votes ###
		###########################

		output_sum = self.sum_up_clause_votes()

        	##############################
		### Calculate Output Value ###
		##############################

		output_value = ((output_sum * (self.max_target-self.min_target))/ self.threshold) + self.min_target

		###########################################
		### Deciding the feedbck to each clause ###
		###########################################

		# Initialize feedback to clauses
		for j in xrange(self.number_of_clauses):
			self.feedback_to_clauses[j] = 0

        	# Type I feedback if target is higher than the predicted value
		if 1.0*(abs(y-output_value))/(self.max_target - self.min_target) > 0.05:
			update_p = 1.0*(abs(y-output_value))/(self.max_target - self.min_target)
		else:
			update_p = 1.0*(abs(y-output_value))/(self.max_target - self.min_target)
		if y > output_value:
			feedback = 0
			for j in xrange(self.number_of_clauses):
				if 1.0*rand()/RAND_MAX < 1.0*(abs(y-output_value))/(self.max_target - self.min_target):
					self.feedback_to_clauses[j] += 1

        	# Type II feedback if target is lower than the predicted value
		elif y < output_value:
			feedback = 1
			for j in xrange(self.number_of_clauses):
				if 1.0*rand()/RAND_MAX < 1.0*(abs(y-output_value))/(self.max_target - self.min_target):
					self.feedback_to_clauses[j] -= 1

		for j in xrange(self.number_of_clauses):
			if self.feedback_to_clauses[j] > 0:

				########################
				### Type I Feedback  ###
				########################

				if self.clause_output[j] == 0:
					for k in xrange(self.number_of_features):
						if 1.0*rand()/RAND_MAX <= 1.0/self.s:
							if self.ta_state[j,k,0] > 1:
								self.ta_state[j,k,0] -= 1

						if 1.0*rand()/RAND_MAX <= 1.0/self.s:
							if self.ta_state[j,k,1] > 1:
								self.ta_state[j,k,1] -= 1

				if self.clause_output[j] == 1:
					for k in xrange(self.number_of_features):
						if X[k] == 1:
							if 1.0*rand()/RAND_MAX <= 1.0*(self.s-1)/self.s:
								if self.ta_state[j,k,0] < self.number_of_states*2:
									self.ta_state[j,k,0] += 1

							if 1.0*rand()/RAND_MAX <= 1.0/self.s:
								if self.ta_state[j,k,1] > 1:
									self.ta_state[j,k,1] -= 1

						elif X[k] == 0:
							if 1.0*rand()/RAND_MAX <= 1.0*(self.s-1)/self.s:
								if self.ta_state[j,k,1] < self.number_of_states*2:
									self.ta_state[j,k,1] += 1

							if 1.0*rand()/RAND_MAX <= 1.0/self.s:
								if self.ta_state[j,k,0] > 1:
									self.ta_state[j,k,0] -= 1

			elif self.feedback_to_clauses[j] < 0:

				#########################
				### Type II Feedback  ###
				#########################
				if self.clause_output[j] == 1:
					for k in xrange(self.number_of_features):
						action_include = self.action(self.ta_state[j,k,0])
						action_include_negated = self.action(self.ta_state[j,k,1])

						if X[k] == 0:
							if action_include == 0 and self.ta_state[j,k,0] < self.number_of_states*2:
								self.ta_state[j,k,0] += 1
						elif X[k] == 1:
							if action_include_negated == 0 and self.ta_state[j,k,1] < self.number_of_states*2:
								self.ta_state[j,k,1] += 1
		#return feedback
	#########################################################
	### Batch Mode Training of Regression Tsetlin Machine ###
	#########################################################

	def fit(self, int[:,:] X, float[:] y, int epochs=1):
		cdef int j, l, epoch
		cdef int example_id
		cdef float target_class
		cdef int[:] Xi
		cdef long[:] random_index
		number_of_examples = len(y)

		Xi = np.zeros((self.number_of_features,), dtype=np.int32)
		random_index = np.arange(number_of_examples)

		for epoch in xrange(epochs):
			np.random.shuffle(random_index)

			for l in xrange(number_of_examples):
				example_id = random_index[l]
				target_class = y[example_id]

				for j in xrange(self.number_of_features):
					Xi[j] = X[example_id,j]
				self.update(Xi, target_class)

		return

	def save_state(self):
		for i in xrange(len(self.ta_state)):
			for j in xrange(len(self.ta_state[i])):
				for k in xrange(len(self.ta_state[i][j])):
					self.saved_ta_state[i][j][k] = self.ta_state[i][j][k]

		for i in xrange(len(self.clause_sign)):
			self.saved_clause_sign[i] = self.clause_sign[i]

		for i in xrange(len(self.clause_output)):
			self.saved_clause_output[i] = self.clause_output[i]

		for i in xrange(len(self.feedback_to_clauses)):
			self.saved_feedback_to_clauses[i] = self.feedback_to_clauses[i]
	def set_state(self):
		for i in xrange(len(self.ta_state)):
			for j in xrange(len(self.ta_state[i])):
				for k in xrange(len(self.ta_state[i][j])):
					self.ta_state[i][j][k] = self.saved_ta_state[i][j][k]
		for i in xrange(len(self.clause_sign)):
			self.clause_sign[i] = self.saved_clause_sign[i]

		for i in xrange(len(self.clause_output)):
			self.clause_output[i] = self.saved_clause_output[i]

		for i in xrange(len(self.feedback_to_clauses)):
			self.feedback_to_clauses[i] = self.saved_feedback_to_clauses[i]
	def __reduce__(self):
		return(type(self), (self.number_of_clauses, self.number_of_features, self.s, self.max_target, self.min_target, self.number_of_states, self.threshold, self.ta_state, self.clause_sign, self.clause_output, self.feedback_to_clauses))
	###
	def get_params(self):
		return self.ta_state, self.clause_sign, self.clause_output, self.feedback_to_clauses

	def set_params(self, ta_state, clause_sign, clause_output, feedback_to_clauses):
		self.ta_state = ta_state
		self.clause_sign = clause_sign
		self.clause_output = clause_output
		self.feedback_to_clauses = feedback_to_clauses

	def upsize_memory(self, new_number_of_states):
		self.number_of_states = new_number_of_states

	cpdef void update_2(self, int[:] X, float y, float advantage, float entropy, float clip):
			cdef int i, j
			cdef int action_include, action_include_negated
			cdef float output_sum
			cdef float output_value
			feedback = -1
			###############################
			### Calculate Clause Output ###
			###############################

			self.calculate_clause_output(X)

			###########################
			### Sum up Clause Votes ###
			###########################

			output_sum = self.sum_up_clause_votes()

				##############################
			### Calculate Output Value ###
			##############################

			output_value = ((output_sum * (self.max_target-self.min_target))/ self.threshold) + self.min_target

			###########################################
			### Deciding the feedbck to each clause ###
			###########################################

			# Initialize feedback to clauses
			for j in xrange(self.number_of_clauses):
				self.feedback_to_clauses[j] = 0

				# Type I feedback if target is higher than the predicted value
			#calculate surrogate loss
			#cant use softmax since it requires softmax between all TMs
			#ratio may not be necessary
			loss = advantage #* ratio
			#loss_1 = advantage * ratio
			ratio = (output_value / y)
			#update_p = 1.0 * advantage * ratio
			#loss = np.min([loss_1, loss_2])
			#print('ratio: ', ratio)

			#print('advantage: ', advantage)
			lr = 0.85
			#lr = 0.85
			#entropy = np.clip(entropy, 0.1, 0.9)
			#entropy = np.clip(entropy, 0.1, 1.0)
			#if entropy < 0.1:
		#		print(entropy)
			if 1.0*(abs(y-output_value))/(self.max_target - self.min_target) > clip:
				update_p = clip
			else:
				update_p = 1.0*(abs(y-output_value))/(self.max_target - self.min_target)
			#print(clip)
			if loss > 0:
				feedback = 0
				for j in xrange(self.number_of_clauses):
					if 1.0*rand()/RAND_MAX < update_p:
					#if 1.0*rand()/RAND_MAX < entropy * 1.0*(abs(y-output_value))/(self.max_target - self.min_target):

					#if 1.0*rand()/RAND_MAX < 1.0*advantage*(output_value/y):#/(self.max_target - self.min_target): //most logical one so far
					#if 1.0*rand()/RAND_MAX < 1.0 * lr * advantage * ratio * entropy:
					#if 1.0*rand()/RAND_MAX < 1 - (y / self.max_target) ** np.abs(advantage): #havent been tested yet
					#if 1.0*rand()/RAND_MAX < 1 - ((y / self.max_target) * entropy) ** np.abs(advantage): #achieved 500
					#if 1.0*rand()/RAND_MAX < 0.1:

					#if 1.0*rand()/RAND_MAX < 1.0*(output_value-y)/(self.max_target - self.min_target):
					#if 1.0*rand()/RAND_MAX < learning_rate * advantage: #1.0*(abs(y-output_value))/(self.max_target - self.min_target):
						self.feedback_to_clauses[j] += 1

				# Type II feedback if target is lower than the predicted value

			elif loss <= 0:
				feedback = 1
				for j in xrange(self.number_of_clauses):
					if 1.0*rand()/RAND_MAX < update_p:
					#if 1.0*rand()/RAND_MAX < 1.0*(abs(y-output_value))/(self.max_target - self.min_target):
					#if 1.0*rand()/RAND_MAX < entropy * 1.0*(abs(y-output_value))/(self.max_target - self.min_target):

					#if 1.0*rand()/RAND_MAX < - 1.0*advantage*(output_value/y):#/(self.max_target - self.min_target): //most logical one so far
					#if 1.0*rand()/RAND_MAX < - 1.0 * lr * advantage * ratio * entropy:
					#if 1.0*rand()/RAND_MAX < (y / self.max_target) ** np.abs(advantage): #havent been tested yet
					#if 1.0*rand()/RAND_MAX <  ((y / self.max_target) * entropy) ** np.abs(advantage): #achieved 500
					#if 1.0*rand()/RAND_MAX <  0.1:
					#if 1.0*rand()/RAND_MAX < update_p:
					#if 1.0*rand()/RAND_MAX < - learning_rate * advantage: #1.0*(abs(y-output_value))/(self.max_target - self.min_target):
						self.feedback_to_clauses[j] -= 1

			for j in xrange(self.number_of_clauses):
				if self.feedback_to_clauses[j] > 0:

					########################
					### Type I Feedback  ###
					########################

					if self.clause_output[j] == 0:
						for k in xrange(self.number_of_features):
							if 1.0*rand()/RAND_MAX <= 1.0/self.s:
								if self.ta_state[j,k,0] > 1:
									self.ta_state[j,k,0] -= 1

							if 1.0*rand()/RAND_MAX <= 1.0/self.s:
								if self.ta_state[j,k,1] > 1:
									self.ta_state[j,k,1] -= 1

					if self.clause_output[j] == 1:
						for k in xrange(self.number_of_features):
							if X[k] == 1:
								if 1.0*rand()/RAND_MAX <= 1.0*(self.s-1)/self.s:
									if self.ta_state[j,k,0] < self.number_of_states*2:
										self.ta_state[j,k,0] += 1

								if 1.0*rand()/RAND_MAX <= 1.0/self.s:
									if self.ta_state[j,k,1] > 1:
										self.ta_state[j,k,1] -= 1

							elif X[k] == 0:
								if 1.0*rand()/RAND_MAX <= 1.0*(self.s-1)/self.s:
									if self.ta_state[j,k,1] < self.number_of_states*2:
										self.ta_state[j,k,1] += 1

								if 1.0*rand()/RAND_MAX <= 1.0/self.s:
									if self.ta_state[j,k,0] > 1:
										self.ta_state[j,k,0] -= 1

				elif self.feedback_to_clauses[j] < 0:

					#########################
					### Type II Feedback  ###
					#########################
					if self.clause_output[j] == 1:
						for k in xrange(self.number_of_features):
							action_include = self.action(self.ta_state[j,k,0])
							action_include_negated = self.action(self.ta_state[j,k,1])

							if X[k] == 0:
								if action_include == 0 and self.ta_state[j,k,0] < self.number_of_states*2:
									self.ta_state[j,k,0] += 1
							elif X[k] == 1:
								if action_include_negated == 0 and self.ta_state[j,k,1] < self.number_of_states*2:
									self.ta_state[j,k,1] += 1
			#return feedback
		#########################################################
		### Batch Mode Training of Regression Tsetlin Machine ###
		#########################################################

	def fit_2(self, int[:,:] X, float[:] y, float[:] advantages, float[:] entropies, float clip=0.1):
		cdef int j, l, epoch
		cdef int example_id
		cdef float target_class
		cdef int[:] Xi
		cdef float advantage
		cdef long[:] random_index
		cdef float entropy
		cdef int epochs = 1

		number_of_examples = len(y)

		Xi = np.zeros((self.number_of_features,), dtype=np.int32)
		random_index = np.arange(number_of_examples)

		for epoch in xrange(epochs):
			np.random.shuffle(random_index)

			for l in xrange(number_of_examples):
				example_id = random_index[l]
				target_class = y[example_id]
				advantage = advantages[example_id]
				entropy = entropies[example_id]
				for j in xrange(self.number_of_features):
					Xi[j] = X[example_id,j]
				self.update_2(Xi, target_class, advantage, entropy, clip)

		return
	cpdef void update_3(self, int[:] X, int feedback_type, float feedback_prob):
			cdef int i, j
			cdef int action_include, action_include_negated
			cdef float output_sum
			cdef float output_value
			feedback = -1
			###############################
			### Calculate Clause Output ###
			###############################



			##############################
			### Calculate Output Value ###
			##############################

			###########################################
			### Deciding the feedbck to each clause ###
			###########################################

			# Initialize feedback to clauses
			for j in xrange(self.number_of_clauses):
				self.feedback_to_clauses[j] = 0

			if feedback_type == 1:
				feedback = 0
				for j in xrange(self.number_of_clauses):
					if 1.0*rand()/RAND_MAX < feedback_prob:
						self.feedback_to_clauses[j] += 1

				# Type II feedback if target is lower than the predicted value

			elif feedback_type == 2:
				feedback = 1
				for j in xrange(self.number_of_clauses):
					if 1.0*rand()/RAND_MAX < feedback_prob:
						self.feedback_to_clauses[j] -= 1

			for j in xrange(self.number_of_clauses):
				if self.feedback_to_clauses[j] > 0:

					########################
					### Type I Feedback  ###
					########################

					if self.clause_output[j] == 0:
						for k in xrange(self.number_of_features):
							if 1.0*rand()/RAND_MAX <= 1.0/self.s:
								if self.ta_state[j,k,0] > 1:
									self.ta_state[j,k,0] -= 1

							if 1.0*rand()/RAND_MAX <= 1.0/self.s:
								if self.ta_state[j,k,1] > 1:
									self.ta_state[j,k,1] -= 1

					if self.clause_output[j] == 1:
						for k in xrange(self.number_of_features):
							if X[k] == 1:
								if 1.0*rand()/RAND_MAX <= 1.0*(self.s-1)/self.s:
									if self.ta_state[j,k,0] < self.number_of_states*2:
										self.ta_state[j,k,0] += 1

								if 1.0*rand()/RAND_MAX <= 1.0/self.s:
									if self.ta_state[j,k,1] > 1:
										self.ta_state[j,k,1] -= 1

							elif X[k] == 0:
								if 1.0*rand()/RAND_MAX <= 1.0*(self.s-1)/self.s:
									if self.ta_state[j,k,1] < self.number_of_states*2:
										self.ta_state[j,k,1] += 1

								if 1.0*rand()/RAND_MAX <= 1.0/self.s:
									if self.ta_state[j,k,0] > 1:
										self.ta_state[j,k,0] -= 1

				elif self.feedback_to_clauses[j] < 0:

					#########################
					### Type II Feedback  ###
					#########################
					if self.clause_output[j] == 1:
						for k in xrange(self.number_of_features):
							action_include = self.action(self.ta_state[j,k,0])
							action_include_negated = self.action(self.ta_state[j,k,1])

							if X[k] == 0:
								if action_include == 0 and self.ta_state[j,k,0] < self.number_of_states*2:
									self.ta_state[j,k,0] += 1
							elif X[k] == 1:
								if action_include_negated == 0 and self.ta_state[j,k,1] < self.number_of_states*2:
									self.ta_state[j,k,1] += 1
			#return feedback
		#########################################################
		### Batch Mode Training of Regression Tsetlin Machine ###
		#########################################################

	def fit_3(self, int[:,:] X, int[:] feedbacks, float[:] feedback_probs, int epochs=1):
		cdef int j, l, epoch
		cdef int example_id
		cdef float target_class
		cdef int[:] Xi
		cdef float advantage
		cdef long[:] random_index
		cdef float entropy

		number_of_examples = len(feedbacks)

		Xi = np.zeros((self.number_of_features,), dtype=np.int32)
		random_index = np.arange(number_of_examples)

		for epoch in xrange(epochs):
			np.random.shuffle(random_index)

			for l in xrange(number_of_examples):
				example_id = random_index[l]
				feedback = feedbacks[example_id]
				feedback_prob = feedback_probs[example_id]
				for j in xrange(self.number_of_features):
					Xi[j] = X[example_id,j]
				self.update_3(Xi, feedback, feedback_prob)

		return