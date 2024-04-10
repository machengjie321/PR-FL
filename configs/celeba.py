EXP_NAME = "CelebA"
L1_GAP = [0.1, 0.1, 0.1, 0.08, 0.05, 0.025, 0.025, 0.01, 0.005]
IMG_DIM = (218, 178)
NUM_FEATURES = 218 * 178
NUM_CLASSES = 2
NUM_TRAIN_DATA = 177457
NUM_TEST_DATA = 22831
NUM_USERS = 9343

NUM_CLIENTS = 10
NUM_LOCAL_UPDATES = 5
CLIENT_BATCH_SIZE = 20
INIT_LR = 0.001

acc_sign = 0.03

n1 = 3
n2 = 3
n3 = 4


# average_download_speed = [20.0]*n1+[6]*n2+[2.4]*n3
# average_upload_speed = [5.0]*n1+[1.5]*n2+[0.6]*n3
# client_density = [1.0]*n1+[0.3]*n2+[0.1]*n3

# average_download_speed = [20.0]*n1+[10]*n2+[10]*n3
# average_upload_speed = [5.0]*n1+[2.5]*n2+[2.5]*n3
# client_density = [1.0]*n1+[0.5]*n2+[0.5]*n3

# average_download_speed = [20.0]*n1+[10]*n2+[6]*n3
# average_upload_speed = [5.0]*n1+[2.5]*n2+[1.5]*n3
# client_density = [1.0]*n1+[0.5]*n2+[0.3]*n3

# average_download_speed = [20.0]*n1+[14]*n2+[10]*n3
# average_upload_speed = [5.0]*n1+[3.5]*n2+[2.5]*n3
# client_density = [1.0]*n1+[0.7]*n2+[0.5]*n3
# list_client_coeff = [1.0]*n1+[client_density[0]/client_density[n1]]*n2+[client_density[0]/client_density[n1+n2]]*n3

average_download_speed = [20.0]*n1+[18]*n2+[16]*n3
average_upload_speed = [5.0]*n1+[4.5]*n2+[4]*n3
client_density = [1.0]*n1+[0.9]*n2+[0.8]*n3

# average_download_speed = [20.0]*n1+[10]*n2+[10]*n3
# average_upload_speed = [5.0]*n1+[2.5]*n2+[2.5]*n3
# client_density = [0.8]*n1+[0.4]*n2+[0.4]*n3

min_density = 0.2

# Conv4
DENSE_TIME = 3.724286518478766
SPARSE_ALL_TIME = 2.66478774077259
SPARSE1_TIME = 1.598415535595268
COEFFICIENTS_SINGLE = [0.0003404870760969184, 6.0777404652307096e-05, 1.7590354819798735e-05, 3.7744218545217262e-06,
                       0.]
# test_num = 0
# train_num = 0
test_num = 6
train_num = 8
patience = 5

asyn_interval = 0.3






SPARSE_TIME = SPARSE1_TIME - sum(COEFFICIENTS_SINGLE)

COMP_COEFFICIENTS = [c * NUM_LOCAL_UPDATES for c in COEFFICIENTS_SINGLE]
# 1MBps = 4e-6 * 2
COMM_COEFFICIENT = 5.561621025626998e-06
TIME_CONSTANT = SPARSE_TIME * NUM_LOCAL_UPDATES

MAX_ROUND = 4001
adaptive_ROUND = 500

# Adaptive pruning config
ADJ_INTERVAL = 20
EVAL_DISP_INTERVAL = 20

IP_MAX_ROUNDS = 1000
IP_ADJ_INTERVAL = ADJ_INTERVAL
IP_DATA_BATCH = 10
IP_THR = 0.1

MAX_INC_DIFF = None
MAX_DEC_DIFF = 0.3

ADJ_THR_FACTOR = 1.5
ADJ_THR_ACC = ADJ_THR_FACTOR / NUM_CLASSES
ADJ_HALF_LIFE = 1000

# Iterative pruning config
NUM_ITERATIVE_PRUNING = 20

# Online algorithm config
MAX_NUM_UPLOAD = 5
