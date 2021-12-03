from utils import *
import time

obs = [-np.pi/4, 5, 20, 10, 3, 3, 30, 10, 5, 3, -np.pi/2, 5, 20, 20, 5, 3, 0, 5, 10, 10, 5, 3, 0, 0, 20, 0, 5, 3, -np.pi/2, 0]
st = time.time()
feature = feature_detection(obs)
print(feature)
print(time.time()-st)