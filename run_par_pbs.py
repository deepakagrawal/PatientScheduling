#!/usr/bin/python
# Example PBS cluster job submission in Python

from subprocess import Popen, PIPE
import sys
import os
import itertools

# If you want to be emailed by the system, include these in job_string:
# PBS -M your_email@address
# PBS -m abe  # (a = abort, b = begin, e = end)

# # Loop over your jobs
# for joint in [0]:
#     for doubly in [0,1]:
#         for policy in ["CP", "SP"]:
#             for hor in [2, 7, 15]:
#                 for th in [1.25, 1.5,1.75]:
#                     for ch in [0, 1]:
#                         # Open a pipe to the qsub command.
#                         proc = Popen(['qsub'], shell = True, stdin = PIPE, stdout = PIPE, stderr = PIPE, close_fds=True)
#                         # Customize your options here
#                         i = 30
#                         job_name = "%02sh%dt%0.2fc%s%d%s" % (policy[:2],hor,th,ch,i,"D" if doubly==1 else "J" if joint==1 else "S")
#                         print(job_name)
#                         walltime = "24:00:00"
#                         processors = "nodes=1:ppn=20"
#                         tempsuffix = "joint" if joint == 1 else "doubly" if doubly == 1 else "simple" + "large"
#                         folder_name = "new3_folder_larger_results"
#                         command = "~/work/anaconda2/envs/py3/bin/python SimulationMain.py -i %d -p 20 -s %d -t %f -c %d -l 1 -o %s -j %d -d %d -n 2 -T 60 -D 30 -f large_dem_cap_loc_data_1.csv -F %s -z %s" % (i,hor, th, ch, policy, joint,doubly,tempsuffix, folder_name)
#                         job_string = """
#             #!/bin/bash
#             #PBS -N %s
#             #PBS -l walltime=%s
#             #PBS -l %s
#             # PBS -l pmem=5gb
#             #PBS -A open
#             #PBS -m a
#             #PBS -M agrawal.deepankur@gmail.com
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
for joint, doubly, policy, hor, th, ch, gm in itertools.product([0, 1], [0], ["DCPP"], [7, 15], [1.25, 1.5, 1.75],
                                                                [0, 1], [1.]):
    # Open a pipe to the qsub command.
    proc = Popen(['qsub'], shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)

    # Customize your options here
    job_name = "%02s%d%.2f%s%s%.2f" % (policy[:2], hor, th, ch, "D" if doubly == 1 else "J" if joint == 1 else "S", gm)
    print(job_name)
    walltime = "24:00:00"
    processors = "nodes=1:ppn=20"
    tempsuffix = "joint" if joint == 1 else "doubly" if doubly == 1 else "simple" + "large"
    folder_name = "rev_DJCP_results_large_dem_cap_loc_data_1"
    command = "~/work/anaconda2/envs/py3/bin/python SimulationMain.py -i 0 40 -p 20 -s %d -t %f -c %d -l 1 -o %s -j %d -d %d -n 2 -T 40 -D 10 --gamma %f --beta 0. -f large_dem_cap_loc_data_1.csv -z %s" % (
    hor, th, ch, policy, joint, doubly, gm, folder_name)
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
    if (sys.version_info > (3, 0)):
        proc.stdin.write(job_string.encode('utf-8'))
    else:
        proc.stdin.write(job_string)
    out, err = proc.communicate()

    # Print your job and the system response to the screen as it's submitted
    print(job_string)
