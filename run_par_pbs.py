#!/usr/bin/python
# Example PBS cluster job submission in Python

from subprocess import Popen, PIPE
import sys
import os

# If you want to be emailed by the system, include these in job_string:
# PBS -M your_email@address
# PBS -m abe  # (a = abort, b = begin, e = end)

# # Loop over your jobs
# for joint in [0]:
#     for doubly in [1]:
#         for policy in ["CP"]:
#             for hor in [2, 7, 15]:
#                 for th in [1.25, 1.75]:
#                     for ch in [0,1]:
#                         # Open a pipe to the qsub command.
#                         proc = Popen(['qsub'], shell = True, stdin = PIPE, stdout = PIPE, stderr = PIPE, close_fds=True)
#
#                         # Customize your options here
#                         i = 30
#                         job_name = "%02sh%dt%0.2fc%s%dS" % (policy[:2],hor,th,ch,i)
#                         print(job_name)
#                         walltime = "2:00:00"
#                         processors = "nodes=1:ppn=1"
#                         tempsuffix = "doublylarge"
#                         folder_name = "new_folder_larger_results"
#                         command = "~/work/anaconda2/envs/py3/bin/python SimulationMain.py -i %d -p 10 -s %d -t %f -c %d -l 1 -o %s -j %d -d %d -n 2 -T 50 -D 20 -f large_dem_cap_loc_data_1.csv -F %s -z %s" % (i,hor, th, ch, policy, joint,doubly,tempsuffix, folder_name)
#                         job_string = """
#             #!/bin/bash
#             #PBS -N %s
#             #PBS -l walltime=%s
#             #PBS -l %s
#             # PBS -l pmem=5gb
#             #PBS -A open
#             #PBS -m abe
#             #PBS -o %s.out
#             #PBS -e %s.err
#             module load gams
#             cd ~/work/PatSchedPang_v2/FairScheduling/gams_try
#             %s""" % (job_name, walltime, processors, job_name, job_name, command)
#                         # Send job_string to qsub
#                         if (sys.version_info >(3,0)):
#                             proc.stdin.write(job_string.encode('utf-8'))
#                         else:
#                             proc.stdin.write(job_string)
#                         out, err = proc.communicate()
#
#                         # Print your job and the system response to the screen as it's submitted
#                         print(job_string)

# Loop over your jobs
for itr in range(10,15): #4
    for policy in ["DP", "DCPP"]:
        for hor in [15]:
            for th in [1.25, 1.75]:
                for ch in [0,1]:
                    # Open a pipe to the qsub command.
                    proc = Popen(['qsub'], shell = True, stdin = PIPE, stdout = PIPE, stderr = PIPE, close_fds=True)

                    # Customize your options here
                    i = itr
                    job_name = "%02sh%dt%0.2fc%s%dS" % (policy[:2],hor,th,ch,i)
                    print(job_name)
                    walltime = "4:00:00"
                    processors = "nodes=1:ppn=1"
                    tempsuffix = "singlelarge"
                    folder_name = "new_folder_larger_results"
                    command = "~/work/anaconda2/envs/py3/bin/python SimulationMain.py -i %d -p 1 -s %d -t %f -c %d -l 1 -o %s -j 0 -d 0 -n 2 -T 50 -D 20 -f large_dem_cap_loc_data_1.csv -F %s -z %s -I" % (i,hor, th, ch, policy, tempsuffix, folder_name)
                    job_string = """
        #!/bin/bash
        #PBS -N %s
        #PBS -l walltime=%s
        #PBS -l %s
        # PBS -l pmem=5gb
        #PBS -A open
        #PBS -m abe
        #PBS -o %s.out
        #PBS -e %s.err
        module load gams
        cd ~/work/PatSchedPang_v2/FairScheduling/gams_try
        %s""" % (job_name, walltime, processors, job_name, job_name, command)
                    # Send job_string to qsub
                    if (sys.version_info >(3,0)):
                        proc.stdin.write(job_string.encode('utf-8'))
                    else:
                        proc.stdin.write(job_string)
                    out, err = proc.communicate()

                    # Print your job and the system response to the screen as it's submitted
                    print(job_string)
