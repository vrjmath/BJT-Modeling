import matplotlib.pyplot as plt

def plot_wf(true_base, predicted_base, true_collector, predicted_collector):
	plt.subplot(1, 2, 1)
	plt.plot(true_collector, label="true")
	plt.plot(predicted_collector, label="predicted")
	plt.title("Collector Current")
	plt.xlabel("Time [ns]")
	plt.ylabel("Normalized Current")
	plt.legend(loc='best')

	plt.subplot(1, 2, 2)
	plt.plot(true_base, label="true")
	plt.plot(predicted_base, label="predicted")
	plt.title("Base Current")
	plt.xlabel("Time [ns]")
	plt.ylabel("Normalized Current")
	plt.legend(loc='best')
	plt.show()
