# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tkinter import filedialog

def find_teb_files(root_folder):
    teb_files = []
    # Regular expression pattern for files named TEB_x, where x is any number
    pattern = re.compile(r'TEBdata_fps_*_')
    
    # Walk through all subfolders and files
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if pattern.match(filename):
                full_path = os.path.join(dirpath, filename)
                teb_files.append(full_path)
    
    return teb_files
def extract_numbers(text):
    pattern = r"[-+]?(?:\d+(?:,\d\d\d)*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    numbers = re.findall(pattern, text)
    return numbers
 
# Example usage
root_folder = filedialog.askdirectory()  # Change this to your folder path
teb_files = find_teb_files(root_folder)
DeltaOR = []
# Print the list of TEB_x files found
for file in teb_files:
    OR = np.loadtxt(file,  delimiter='\t', usecols=(0), unpack=True)
    file_base = os.path.basename(file)
    DeltaOR.append(max(OR)-min(OR))
    segments = file_base.split('_')
    fps = float(segments[2])
    plt.plot(np.linspace(0,len(OR)/fps,len(OR)) ,OR)
    print(fps)
plt.xlabel('time [s]')
plt.ylabel('normalized retardation')
plt.show()

for file in teb_files:

    OR = np.loadtxt(file,  delimiter='\t', usecols=(0), unpack=True)
    file_base = os.path.basename(file)
    segments = file_base.split('_')
    fps = float(segments[2])
    OR = (OR-min(OR))
    OR = OR/max(OR)
    plt.plot(np.linspace(0,len(OR)/fps,len(OR)) ,OR)
    print(fps)

plt.xlabel('time [s]')
plt.ylabel('normalized retardation')
TEB_simulated_eff = np.loadtxt('C:/Users/marc3/OneDrive/Documents/INTERNSHIP-PHD/MODEL RHAB/TEB_corr.txt')
#plt.plot( np.linspace(0,5, len(TEB_simulated_eff) ),TEB_simulated_eff)
plt.show()



# %%
plt.plot([0.5, 0.25, 1/8, 1/16, 1/32],[ DeltaOR[1], DeltaOR[3], DeltaOR[4], DeltaOR[0], DeltaOR[2] ],'.')
plt.show()
# %%
plt.errorbar([0.5,0.25,1/8,1/16,1/32],[8.024,5.96,2.29,0.37,0.16],[0.96,0.47,0.073,0.008,0.029])

# %%
