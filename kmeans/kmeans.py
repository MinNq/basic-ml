import numpy as np
from scipy.linalg import norm


class KMeans:

	def __init__(self, n_clusters, init = 'kmeans++', max_iter = 300):

		self.n_clusters = n_clusters
		self.max_iter = max_iter
		self.init = init

	def fit(self, X):
		
		'''
		Compute k-means clustering.

		Parameter:
		- X: array of shape (n_samples, n_features)
		'''

		self.X = np.array(X)
		self.n_samples, self.n_features = self.X.shape

		# initializing centers
		self.initialize()

		# initializing iteration
		self.n_iter_ = 0

		self.labels_log = []
		self.inertia_log = []

		while True:

			self.n_iter_ += 1
			
			# assignment step
			self.assign()

			# update step
			self.update()

			# stopping criteria
			if self.n_iter_ > 1:
				reached_max_iter = self.n_iter_ == self.max_iter
				converged = self.inertia_log[self.n_iter_ - 1] == self.inertia_log[self.n_iter_ - 2]

				if converged or reached_max_iter:
					break

	def initialize(self):

		if self.init == 'kmeans++':
			
			self.cluster_centers_ = np.zeros((self.n_clusters, self.n_features))
			self.cluster_centers_[0] = np.random.permutation(self.X)[0]

			for index in range(1, self.n_clusters):

				dist = []
				for point in self.X:
					distances = [norm(point - center) for center in self.cluster_centers_]
					dist.append(min(distances)**2)
				dist = np.divide(dist, sum(dist))

				choice = np.random.choice(self.n_samples, p = dist)
				self.cluster_centers_[index] = self.X[choice]

		if self.init == 'random':
			
			self.cluster_centers_ = np.random.permutation(self.X)[:self.n_clusters]


	def assign(self):

		self.labels_ = np.zeros((self.n_samples,))
		self.labels_log.append(np.zeros((self.n_samples,)))
		self.inertia_ = 0
		self.inertia_log.append(0)

		self.cluster_sums_ = np.zeros((self.n_clusters, self.n_features))
		self.cluster_sizes_ = np.zeros((self.n_clusters,))
		
		for index, point in enumerate(self.X):
			distances = [norm(point - center) for center in self.cluster_centers_]
			self.labels_[index] = np.argmin(distances)
			self.labels_log[self.n_iter_ - 1][index] = np.argmin(distances)

			# loss function
			self.inertia_ += min(distances)**2
			self.inertia_log[self.n_iter_ - 1] += min(distances)**2

			# for computing centers
			self.cluster_sums_[np.argmin(distances)] += point
			self.cluster_sizes_[np.argmin(distances)] += 1

	def update(self):

		nom = self.cluster_sums_.T
		denom = self.cluster_sizes_
		self.cluster_centers_ = np.divide(nom, denom).T

	def predict(self, X):

		X = np.array(X)
		labels_ = np.zeros((X.shape[0],))
		for index, point in enumerate(X):
			distances = [norm(point - center) for center in self.cluster_centers_]
			labels_[index] = np.argmin(distances)

		return labels_