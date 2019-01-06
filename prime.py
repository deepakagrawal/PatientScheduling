from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# def Pi(num_steps):
#     step = 1.0/num_steps
#     sum = 0
#     for i in range(num_steps):
#         x = (i+0.5)*step
#         sum += 4.0/(1.0+x**2)
#     pi = step*sum
#     return pi



# print("The pi estimate from process %d/%d is %s" %(rank, size, Pi(100*(rank+1))))
a = 100;
a = a+100;

data = [(rank+1)**2]
data = comm.gather(data, root = 0)
if rank==0:
    print(data)

print(a)


# from time import sleep
# from jug import TaskGenerator
#
# @TaskGenerator
# def is_prime(n):
#     sleep(1.)
#     for j in range(2, n-1):
#         if (n %j) == 0:
#             return False
#     return True
#
# @TaskGenerator
# def count_primes(ps):
#     return sum(ps)
#
# @TaskGenerator
# def write_output(n):
#     output = open('output.txt', 'wt')
#     output.write("Found {0} primes <= 100.\n".format(n))
#     output.close()
#
# primes100 = []
# for n in range(2,101):
#     primes100.append(is_prime(n))
#
# n_primes = count_primes(primes100)
# write_output(n_primes)