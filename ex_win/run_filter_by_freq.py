import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from s_signal import components_selection_one_signal, btbp_filter_signal
import matplotlib.pyplot as plt

from s_data_ex_win import load_data_win
from IPython.display import display

rdic, dps = load_data_win()

#sig_name = dps[1]
sig_name = "aX_t_W00133_exp01_user01_act02.txt"
sig_raw = rdic[sig_name]
display(sig_name)
#display(Raw_dic)

results = components_selection_one_signal(sig_raw)
sig_total = results[0]
sig_dc = results[1]
sig_body = results[2]
sig_noise = results[3]

sig_bp = btbp_filter_signal(sig_raw)


fig = plt.figure(figsize=(16, 9))

time = [1 / float(50) * i for i in range(len(sig_raw))]

# ploting each signal component
_ = plt.plot(time, sig_raw, label="sig_raw")
_ = plt.plot(time, sig_dc, label="sig_dc")
_ = plt.plot(time, sig_body, label="sig_body")
_ = plt.plot(time, sig_noise, label="sig_noise")
_ = plt.plot(time, sig_bp, label="sig_bp")

# Set the figure info defined earlier
#_ = plt.ylabel(figure_Ylabel)  # set Y axis info
# Set X axis info (same label in all cases)
_ = plt.xlabel('Time in seconds (s)')
_ = plt.title("signals")  # Set the title of the figure

# localise the figure's legends
_ = plt.legend(loc="upper left")  # upper left corner

# showing the figure
plt.show()
