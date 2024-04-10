EXP_NAME = "CIFAR10"
L1_GAP = [0.1, 0.1, 0.1, 0.08, 0.05, 0.025, 0.025, 0.01, 0.005]
NUM_FEATURES = 3 * 32 * 32
NUM_CLASSES = 10
NUM_TRAIN_DATA = 50000
NUM_TEST_DATA = 10000

NUM_CLIENTS = 10
NUM_LOCAL_UPDATES = 5
CLIENT_BATCH_SIZE = 20
INIT_LR = 0.25

acc_sign = 0.03

EVAL_DISP_INTERVAL = 20
n1 = 2
n2 = 2
n3 = 2
n4 = 4
average_download_speed = [20.0,18.0]+[12,10]+[6.0,4.0]+[2.5,2.0,2.0,1.5]
average_upload_speed = [5.0,4.0]+[3.0,2.5]+[1.5,1.0]+[0.6,0.50,0.50,0.4]
client_density = [1.0]*n1+[0.6]*n2+[0.3]*n3+[0.1]*n4

holistic_coeff = 10
list_client_coeff = [1.0]*n1+[client_density[0]/client_density[n1]]*n2+[client_density[0]/client_density[n1+n2]]*n3
min_density = 0.05


# VGG-11
DENSE_TIME = 31.514276721399803
SPARSE_ALL_TIME = 31.08082229309948
SPARSE_TIME = 17.36990105989628

COEFFICIENTS_SINGLE = [0., 8.95507e-6, 2.495288e-6, 2.780686e-6, 1.024265e-6, 1.277773e-6, 1.843831e-6,
                       8.066104e-7, 5.145334e-7, 2.430023e-7, 0.]
COMP_COEFFICIENTS = [c * NUM_LOCAL_UPDATES for c in COEFFICIENTS_SINGLE]
# 1MBps = 4e-6 * 2
COMM_COEFFICIENT = 5.561621025626998e-06  # 5.18405073e-6
TIME_CONSTANT = SPARSE_TIME * NUM_LOCAL_UPDATES

MAX_ROUND = 10001
adaptive_ROUND = 5000
test_num = 4
train_num = 4
patience = 10
asyn_interval = 1
# Adaptive pruning config
ADJ_INTERVAL = 50
model_size = 37.203980445861816

IP_MAX_ROUNDS = 1000
IP_ADJ_INTERVAL = ADJ_INTERVAL
IP_DATA_BATCH = 10
IP_THR = 0.1

ADJ_THR_FACTOR = 1.5
ADJ_THR_ACC = ADJ_THR_FACTOR / NUM_CLASSES

# Variables
MAX_INC_DIFF = None
MAX_DEC_DIFF = 0.3

LR_HALF_LIFE = 5500
ADJ_HALF_LIFE = 5500

# Iterative pruning config
NUM_ITERATIVE_PRUNING = 20

# Online algorithm config
MAX_NUM_UPLOAD = 5
