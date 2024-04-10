EXP_NAME = "DBLP"

IMG_DIM = (28, 28)
NUM_CLASSES = 4
NUM_TRAIN_DATA = 11258
vocabulary_size=  1995

save_interval = 10


NUM_CLIENTS = 10   # Set the number of client
NUM_LOCAL_UPDATES = 10
CLIENT_BATCH_SIZE = 64
INIT_LR = 0.25

ADJ_INTERVAL = 10
EVAL_DISP_INTERVAL = 10




# Conv2
DENSE_TIME = 2.2486449218005875  # 10 times
SPARSE_ALL_TIME = 2.898095353249955  # all params are in, but sparse form
SPARSE_TIME = 0.2492789765
COMP_COEFFICIENTS_SINGLE = [1.782308e-7, 1.782308e-7, 1.782308e-7, 1.782308e-7]
COMP_COEFFICIENTS = [coeff * NUM_LOCAL_UPDATES for coeff in COMP_COEFFICIENTS_SINGLE]
COMM_COEFFICIENT = 2.61621025626998e-05
C_COMP = SPARSE_TIME * NUM_LOCAL_UPDATES
C_COMM = 0.
TIME_CONSTANT = C_COMP + C_COMM

TO_SPARSE_THR = 0.9

MAX_INC_DIFF = None
MAX_DEC_DIFF = 0.3

ADJ_THR_FACTOR = 1.5
ADJ_THR_ACC = ADJ_THR_FACTOR / NUM_CLASSES
ADJ_HALF_LIFE = 400

MAX_ROUND = 401
MAX_ROUND_CONVENTIONAL_FL= 401
MAX_ROUND_ADAPTIVE=401

hidden_size = (100, 100)

