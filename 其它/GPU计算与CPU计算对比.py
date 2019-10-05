import torch
import time

###CPU
start_time = time.time()
a = torch.rand(4000, 4000)
for _ in range(1000):
    a += a
elapsed_time = time.time() - start_time

print('CPU time = ', elapsed_time)

###GPU
start_time = time.time()
b = torch.rand(4000, 4000).cuda()
for _ in range(1000):
    b += b
elapsed_time = time.time() - start_time

print('GPU time = ', elapsed_time)

# 实验结果
# CPU time =  403.8645272254944
# GPU time =  24.993326425552368
# 循环10000次  ones 4000
# Process finished with exit code 0