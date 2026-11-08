import sys; sys.path.insert(0,'.')
from common import *
print("torch", torch.__version__, "threads", torch.get_num_threads())
for dev in (['cpu','mps'] if torch.backends.mps.is_available() else ['cpu']):
    sync = torch.mps.synchronize if dev=='mps' else None
    for (M,K,N) in [(13634,2048,2048),(13634,86,2048),(6635,2120,2048),(13634,300,300),(4096,4096,4096)]:
        A=torch.randn(M,K,device=dev); B=torch.randn(N,K,device=dev)
        mn,_=timeit(lambda: torch.nn.functional.linear(A,B), sync=sync)
        print(f"{dev} linear [{M}x{K}] @ [{K}x{N}]: {mn*1e3:8.2f} ms  {2*M*K*N/mn/1e9:8.1f} GFLOP/s")
