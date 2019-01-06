from mpi4py.futures import MPIPoolExecutor

executor = MPIPoolExecutor(max_workers=2)
future = executor.submit(pow, 2,3)
print(future.result())