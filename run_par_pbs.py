#!/usr/bin/python
# Example PBS cluster job submission in Python

from subprocess import Popen, PIPE
import sys
import os

# If you want to be emailed by the system, include these in job_string:
# PBS -M your_email@address
# PBS -m abe  # (a = abort, b = begin, e = end)

# Loop over your jobs
for policy in ["DCPP"]:
    for hor in [2]:
        for th in [1.25,1.75]:
            for ch in [0]:
                # Open a pipe to the qsub command.
                proc = Popen(['qsub'], shell = True, stdin = PIPE, stdout = PIPE, stderr = PIPE, close_fds=True)

                # Customize your options here
                job_name = "%s_h%d_t%0.1f_c%s" % (policy[1],hor,th,ch)
                walltime = "24:00:00"
                processors = "nodes=1:ppn=20"
                command = "~/work/anaconda2/envs/py3/bin/python SimulationMain.py -i 20 -p 20 -s %d -t %f -c %d,%d -l 1,2 -o %s -j 1 -d 0 -n 2" % (hor, th, ch, ch+1,policy)

                job_string = """
    #!/bin/bash
    #PBS -N %s
    #PBS -l walltime=%s
    #PBS -l %s
    # PBS -l pmem=5gb
    #PBS -A open
    #PBS -m abe
    #PBS -M dua143@psu.edu
    #PBS -o %s.out
    #PBS -e %s.err
    module load gams
    cd ~/work/PatSchedPang_v2/FairScheduling/gams_try
    %s""" % (job_name, walltime, processors, job_name, job_name, command)

                # print('start writing')
                # test_file = open(job_name, "w")
                # test_file.write(job_string)
                # test_file.close()
                # Send job_string to qsub
                if (sys.version_info >(3,0)):
                    proc.stdin.write(job_string.encode('utf-8'))
                else:
                    proc.stdin.write(job_string)
                out, err = proc.communicate()

                # Print your job and the system response to the screen as it's submitted
                print(job_string)
