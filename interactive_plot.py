import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
import os
import glob
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Sample dataset
x_data = np.linspace(0, 10, 100)
y_data = 3 * np.sin(1.5 * x_data + 0.5) + np.random.normal(0, 0.5, size=x_data.shape)

# Initial parameter values for the model function
a0 = 1.0
alpha0 = 1.0
D0 = 1.0
c0 = 0.0
def st_exponential(x, a,D,c,alpha): # stretched exponential
     return a * np.exp( -((6*D) * x)**alpha) + c 


# Create the plot
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.35)

# Plot dataset
scatter = ax.scatter(x_data, y_data, label='Data', color='gray')

# Plot initial model
model_line, = ax.plot(x_data, st_exponential(x_data, a0, D0, c0, alpha0), label='Model', color='red')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()

# Define sliders for parameters
a_slider_ax = fig.add_axes([0.25, 0.25, 0.65, 0.03])
D_slider_ax = fig.add_axes([0.25, 0.20, 0.65, 0.03])
c_slider_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03])
alpha_slider_ax = fig.add_axes([0.25, 0.10, 0.65, 0.03])

a_slider = Slider(a_slider_ax, 'a', 0.1, 100, valinit=a0)
D_slider = Slider(D_slider_ax, 'D', 0.1, 10, valinit=D0)
c_slider = Slider(c_slider_ax, 'c', 0, 100, valinit=c0)
alpha_slider = Slider(alpha_slider_ax, 'alpha', 0.0001, 1, valinit=a0)
# Update function for sliders
def update(val):
    a = a_slider.val
    D = D_slider.val
    c = c_slider.val
    alpha = alpha_slider.val

    model_line.set_ydata(st_exponential(x_data, a, D, c, alpha))
    fig.canvas.draw_idle()

# Connect sliders to update function
a_slider.on_changed(update)
D_slider.on_changed(update)
c_slider.on_changed(update)
alpha_slider.on_changed(update)
plt.show()
