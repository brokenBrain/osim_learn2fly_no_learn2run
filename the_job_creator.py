####################
#
#	This file creates the directory structure for exploring
#	parameter spaces
#
####################

import os
import re
import sys
import stat
import shutil

param_vals = re.split("_", sys.argv[2])
exploration_title = sys.argv[1]
num_tries = int(sys.argv[3])

debug_dir = "../outputs/" + exploration_title
if not os.path.exists(debug_dir):
	os.makedirs(debug_dir)

shutil.copy("ppo_approach.py", debug_dir)

for i in range(len(param_vals)):
	curr_debug_dir = debug_dir + "/" + param_vals[i]
	if not os.path.exists(curr_debug_dir):
		os.makedirs(curr_debug_dir)	
	for j in range(num_tries):
		if not os.path.exists(curr_debug_dir + "/" + str(j)):
			os.makedirs(curr_debug_dir + "/" + str(j))
