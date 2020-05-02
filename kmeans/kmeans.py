import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('ggplot')


class KMeans:

	def __init__(self, n_clusters, max_iter = 300):

		self.n_clusters = n_clusters
		self.max_iter = max_iter

	def fit(self, X):
		
		'''
		Do k-means clustering.

		Parameter:
		- X: array of shape (n_samples, n_features)
		'''

		self.X = np.array(X)
		n_samples, n_features = self.X.shape

		# initializing centers
		self.cluster_centers_ = np.random.permutation(self.X)[:self.n_clusters]

		# initializing iteration and loss log
		self.n_iter_ = 0
		self.labels_ = []
		self.inertia_ = []

		while True:

			self.n_iter_ += 1
			self.labels_.append(np.zeros((n_samples,)))
			self.inertia_.append(0)

			self.cluster_sums_ = np.zeros((self.n_clusters, n_features))
			self.cluster_sizes_ = np.zeros((self.n_clusters,))
			
			# assignment step
			for index, point in enumerate(self.X):
				distances = [norm(point - center)**2 for center in self.cluster_centers_]
				self.labels_[self.n_iter_ - 1][index] = np.argmin(distances)

				# loss function
				self.inertia_[self.n_iter_ - 1] += min(distances)

				# for computing centers
				self.cluster_sums_[int(self.labels_[self.n_iter_ - 1][index])] += point
				self.cluster_sizes_[int(self.labels_[self.n_iter_ - 1][index])] += 1

			# update step
			nom = self.cluster_sums_.T
			denom = self.cluster_sizes_
			self.cluster_centers_ = np.divide(nom, denom).T

			# stopping criteria
			if self.n_iter_ > 1:
				reached_max_iter = self.n_iter_ == self.max_iter
				converged = self.inertia_[self.n_iter_ - 1] == self.inertia_[self.n_iter_ - 2]

				if converged or reached_max_iter:
					break


	def predict(self, X):

		X = np.array(X)
		labels_ = np.zeros((X.shape[0],))
		for index, point in enumerate(X):
			distances = [norm(point - center)**2 for center in self.cluster_centers_]
			labels_[index] = np.argmin(distances)

		return labels_[index]


	def animate(self):
		
		'''
		Animate cluster assigning and loss function through iterations.
		Only for models with 2 features.
		'''

		# preparing figure and axes
		fig, axs = plt.subplots(1, 2, figsize = (9, 4.5))
		plt.subplots_adjust(top = .8, bottom = .15, wspace = .3)
    
		axs[0].set_xlabel('Feature 1')
		axs[0].set_ylabel('Feature 2')

		axs[1].set_xlim(-0.05*self.n_iter_ + 1, 1.05*self.n_iter_)
		axs[1].set_ylim(self.inertia_[self.n_iter_ - 1] - 0.05*self.inertia_[0], 
                    	1.05*self.inertia_[0])
		axs[1].set_xlabel('Iteration')
		axs[1].set_ylabel('Within-cluster sum of squares')

		# initializing subplots
		scatters = []

		for _ in range(self.n_clusters):
			scatters.append(axs[0].scatter(self.X[:, 0],self.X[:, 1], alpha = .5, s = 15))

    	# animation update function
		def show_animation(frame):

			plt.suptitle('$k$-means Clustering (Iteration {}/{})'.format(frame + 1, 
							self.n_iter_), y = .9, size = 'x-large')
	
			for index in range(self.n_clusters):
				in_cluster = np.array([point for index2, point in enumerate(self.X) if 
								index == self.labels_[frame][index2]])
				scatters[index].set_offsets(np.c_[in_cluster[:,0], in_cluster[:,1]])

			axs[1].plot(range(1, frame + 2), self.inertia_[:frame + 1], color = 'blue')


		anim = FuncAnimation(fig, show_animation, interval = 200, 
								frames = range(self.n_iter_))

		anim.save('kmeans.gif', writer = 'imagemagick', dpi = 200)


'''
Let's test ~
'''

np.random.seed(0)

# generating data
X = []
means = [[1, 1], [-.2, -.2], [1, -1], [-1, 1]]
for index in range(4):
	cov = [[.01*index + .08, 0],[0, .01*index + .08]]
	X.extend(np.random.multivariate_normal(means[index], cov, 175))

k = KMeans(n_clusters = 4)

k.fit(X)
k.animate()