from tensorboardX import SummaryWriter
import numpy as np

a = np.array([1, 2, 3])
print(a)

writer = SummaryWriter()
writer.add_scalar("test", 1, 0)
writer.close()
