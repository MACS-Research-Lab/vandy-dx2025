import pandas as pd

import time
import sys
import os

from VandyDiagnosisSystem import VandyDiagnosisSystem

TIMEOUT = 0.1

## Create diagnosis system class

# Create diagnosis system. 
EDS = VandyDiagnosisSystem() # change to your class here

## Initialize diagnosis system
# The diagnosis system parameter file is created by calling EDS.Train()
# You can initialize the diagnosis system by loading a parameter file
EDS.Initialize()

# Load test data
if len(sys.argv) != 2:
    print("Usage: python run_diagnoser.py <input_file>")
    sys.exit(1)

input_file = sys.argv[1]
test_data = pd.read_csv(input_file, sep=',')

# Log diagnosis output
output_file = os.path.join('results', 'output_' + os.path.basename(input_file))

filehandler = open(output_file, 'w')
filehandler.write('sample_time,computation_time,detection,fpic_rank,fpim_rank,fwaf_rank,fiml_rank,fx_rank\n');

for time_idx in range(len(test_data)):
    t = time.time()

    # Feed sample to diagnosis system and return diagnosis output
    detection, isolation = EDS.Input(test_data.iloc[time_idx,:].to_frame().transpose())
    
    elapsed = time.time() - t
    
    # Log diagnosis output
    filehandler.write('%f,%f,%d,%f,%f,%f,%f,%f \n' % (test_data['time'][time_idx], elapsed, detection[0], isolation[0,0], isolation[0,1], isolation[0,2], isolation[0,3], isolation[0,4]))

    # Check if solution is too slow
    if elapsed > TIMEOUT and time_idx > 5:
        print(f"Timeout at time index {time_idx} ({test_data['time'][time_idx]})")
        break

    # Print progress
    if time_idx % 1000 == 0:
        print('.', end="")
    
# Close logger
filehandler.close()