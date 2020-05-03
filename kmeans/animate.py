import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from kmeans import KMeans

#plt.style.use('dark_background')
plt.style.use('ggplot')


np.random.seed(0)

# generating data
X = []
means = [[.7, .8], [-.2, -.2], [1, -1], [-1, 1]]
for index in range(4):
	cov = [[.01*index + .08, 0],[0, .01*index + .08]]
	X.extend(np.random.multivariate_normal(means[index], cov, 125))

k = KMeans(n_clusters = 4)

k.fit(X)

'''
animation
'''

# preparing figure and axes
fig, axs = plt.subplots(1, 2, figsize = (9, 4.5))
plt.subplots_adjust(top = .8, bottom = .15, wspace = .3)

axs[0].set_xlabel('Feature 1')
axs[0].set_ylabel('Feature 2')
axs[0].grid(None)

axs[1].set_xlim(-0.05*k.n_iter_ + 1, 1.05*k.n_iter_)
axs[1].set_ylim(k.inertia_log[k.n_iter_ - 1] - 0.05*k.inertia_log[0], 
				1.05*k.inertia_log[0])
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Within-cluster sum of squares')
#axs[1].grid(color = 'dimgray')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# initializing subplots
scatters = []

for index in range(k.n_clusters):
	scatters.append(axs[0].scatter(k.X[:, 0], k.X[:, 1], alpha = .2, s = 15, 
					color = colors[index]))
	scatters.append(axs[0].scatter([],[], alpha = .8, s = 45, 
					color = colors[index]))

# animation update function
def show_animation(frame):

	plt.suptitle('$k$-means Clustering (Iteration {}/{})'.format(frame + 1, 
				k.n_iter_), y = .9, size = 'x-large')
	
	for index in range(k.n_clusters):
		in_cluster = np.array([point for index2, point in enumerate(k.X) if
							index == k.labels_log[frame][index2]])
		scatters[2*index].set_offsets(np.c_[in_cluster[:,0], in_cluster[:,1]])

		mean = np.mean(in_cluster, axis = 0)
		scatters[2*index + 1].set_offsets(np.c_[mean[0], mean[1]])

	axs[1].plot(range(1, frame + 2), k.inertia_log[:frame + 1], lw = 1, 
				color = colors[0])


anim = FuncAnimation(fig, show_animation, interval = 200, 
					frames = range(k.n_iter_))

anim.save('kmeans.gif', writer = 'imagemagick', dpi = 200)