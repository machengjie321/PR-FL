EXP_NAME = "MNIST"
L1_GAP = [0.1, 0.1, 0.1, 0.08, 0.05, 0.025, 0.025, 0.01, 0.005]
IMG_DIM = (28, 28)
NUM_FEATURES = 28 * 28
NUM_CLASSES = 10
NUM_TRAIN_DATA = 60000
NUM_TEST_DATA = 10000
model_size = 0.084808349609375

n1 = 2
n2 = 2
n3 = 2
n4 = 4
average_download_speed = [20.0,18.0]+[12,10]+[6.0,4.0]+[2.5,2.0,2.0,1.5]
average_upload_speed = [5.0,4.0]+[3.0,2.5]+[1.5,1.0]+[0.6,0.50,0.50,0.4]
client_density = [1.0]*n1+[0.6]*n2+[0.3]*n3+[0.1]*n4

holistic_coeff = 10
list_client_coeff = [1.0]*n1+[client_density[0]/client_density[n1]]*n2+[client_density[0]/client_density[n1+n2]]*n3
min_density = 0.08

acc_sign = 0.01

NUM_CLIENTS = 10     # Set the number of client
NUM_LOCAL_UPDATES = 5
CLIENT_BATCH_SIZE = 5
INIT_LR = 0.01

ADJ_INTERVAL = 50
EVAL_DISP_INTERVAL = 50

IP_MAX_ROUNDS = 1000
IP_ADJ_INTERVAL = ADJ_INTERVAL
IP_DATA_BATCH = 10
IP_THR = 0.1




# Conv2
DENSE_TIME = 2.2486449218005875  # 10 times
SPARSE_ALL_TIME = 2.898095353249955  # all params are in, but sparse form
SPARSE_TIME = 1.2492789765
COMP_COEFFICIENTS_SINGLE = [0., 7.098850e-6, 1.927325e-7, 1.782308e-7]
COMP_COEFFICIENTS = [coeff * NUM_LOCAL_UPDATES for coeff in COMP_COEFFICIENTS_SINGLE]
COMM_COEFFICIENT = 5.561621025626998e-06
C_COMP = SPARSE_TIME * NUM_LOCAL_UPDATES
C_COMM = 0.
TIME_CONSTANT = C_COMP + C_COMM

TO_SPARSE_THR = 0.9

MAX_INC_DIFF = None
MAX_DEC_DIFF = 0.3

ADJ_THR_FACTOR = 1.5
ADJ_THR_ACC = ADJ_THR_FACTOR / NUM_CLASSES
ADJ_HALF_LIFE = 7000

MAX_ROUND = 10001

MAX_ROUND_CONVENTIONAL_FL=1000
MAX_ROUND_ADAPTIVE=10000
# test_num = 4
# train_num = 4
test_num = 0
train_num = 0
patience = 15
asyn_interval = 0.5
# Iterative pruning config
NUM_ITERATIVE_PRUNING = 20

# Online algorithm config
MAX_NUM_UPLOAD = 5
