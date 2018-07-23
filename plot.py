import utils
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

steps = utils.load_obj("AcademySave/log/steps_approx")
steps = [(np.sum(x[1]) / len(x[1])) for x in steps]

# xnew = np.linspace(0,len(steps),100000) #300 represents number of points to make between T.min and T.max

x = [i for i in range(len(steps))]
steps_smooth = interp1d(x, steps, kind="cubic")
xnew = np.linspace(0, 99, num=20, endpoint=True)
# the histogram of the data
plt.plot(xnew, steps_smooth(xnew))

plt.xlabel('Episodes progress, %')
plt.ylabel('Steps')
plt.title(r'Average game score of Q-table: episodes=200000')
plt.axis([0, 100, 0, 220])
plt.xticks([x*5 for x in range(20)])
plt.grid(True)

plt.show()
