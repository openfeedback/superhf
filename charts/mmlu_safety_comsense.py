import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as mtick
import cha


# Call the set_plot_style function to set the seaborn theme and other styling parameters
set_plot_style()

# Set the width of the bars
barWidth = 0.15



# Set the data
evaluations = ['MMLU', 'Common Sense', 'Safety']

Alpaca = np.array([0.34394035913428933, 0.6292318320364014, 0.6626076935240235]) * 100
Llama = np.array([0.3119035803354221, 0.6099863078256774, 0.540685696149524]) * 100
RLHF = np.array([0.3369854258823811, 0.6300493429774198, 0.6443659103785468]) * 100
SuperHF = np.array([0.34461623820506443, 0.6287948494238742, 0.6705171927145258]) * 100

# Subtract Llama values from others
Alpaca -= Llama
RLHF -= Llama
SuperHF -= Llama
Llama -= Llama - 0.2

# Standard errors
alpaca_7b_stderr = np.array([0.035145507173984965, 0.011640221046600886, 0.006787403716561199]) * 100
llama_7b_stderr = np.array([0.03437496210298489, 0.01170265180130513, 0.006958014676005136]) * 100
RLHF_stderr = np.array([0.03489073357486305, 0.011616057348211164, 0.00673004486012271]) * 100
SuperHF_stderr = np.array([0.035134698183086434, 0.011637326683180068, 0.006792445694112547]) * 100

# Set position of bar on X axis
r1 = np.arange(len(Alpaca))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.bar(r1, Llama, color=model_type_to_palette_color('pretrained'), width=barWidth, edgecolor='grey', label='Llama (7B)', yerr=llama_7b_stderr, capsize=3)
plt.bar(r2, Alpaca, color=model_type_to_palette_color('instruct'), width=barWidth, edgecolor='grey', label='Alpaca (7B)', yerr=alpaca_7b_stderr, capsize=3)
plt.bar(r3, RLHF, color=model_type_to_palette_color('RLHF'), width=barWidth, edgecolor='grey', label='RLHF', yerr=RLHF_stderr, capsize=3)
plt.bar(r4, SuperHF, color=model_type_to_palette_color('SuperHF'), width=barWidth, edgecolor='grey', label='SuperHF', yerr=SuperHF_stderr, capsize=3)


# Add xticks on the middle of the group bars
plt.xlabel('Evaluations', fontweight='bold')
plt.ylabel('Accuracy Normalized (%)', fontweight='bold')
plt.xticks([r + barWidth*1.5 for r in range(len(Alpaca))], evaluations)

# Adjust y-axis limits to show negative bars
plt.ylim(min(np.min(Alpaca), np.min(RLHF), np.min(SuperHF)) - 10, max(np.max(Alpaca), np.max(RLHF), np.max(SuperHF)) + 10)

# Create a function to format y-axis as percentages
fmt = '%.0f%%'
yticks = mtick.FormatStrFormatter(fmt)
plt.gca().yaxis.set_major_formatter(yticks)
plt.title('Downstream Evaluations')
plt.legend()

plt.savefig('plot.png', dpi=300)

plt.show()
