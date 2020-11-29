import numpy as np

# Paths
DATA_DIR = r'./data/'
DATA_OCT = DATA_DIR+'2019-Oct.csv'
DATA_NOV = DATA_DIR+'2019-Nov.csv'
EXPORT_DIR = r'./export/'
UX_CONSTANTS = DATA_DIR+r'ux_constants.csv'
NEW_USER_ID = DATA_DIR+r'new_user_id.csv'
NEW_PRODUCT_ID = DATA_DIR+r'new_product_id.csv'

# Paths for the generalised solution
DATASET_5M = r"../_data/eCommerce-cosmetics/"
DATASET_SALE = r"../_data/eCommerce-purchase-history/"

LOG_DIR = r'C:\TensorLogs'
ALL_DATA_PATH = DATA_DIR+r'uxm.npz'
TRAIN_DATA_PATH = DATA_DIR+r'uxm_train.npz'
VAL_DATA_PATH = DATA_DIR+r'uxm_val.npz'
TEST_DATA_PATH = DATA_DIR+r'uxm_test.npz'

# Minimum number of events a user needs to have before being included in the dataset
EVENT_THRESHOLD = 5

# constants for metrics
DEFAULT_K = 10 
DEFAULT_THRESHOLD = 10

# Train / Validation / Test split thresholds
VAL_THRESHOLD = 0.7
TEST_THRESHOLD = VAL_THRESHOLD+(1-VAL_THRESHOLD)/2

# Default column names
USECOLS = ["event_type","product_id","user_id"]



USER = DEFAULT_USER_COL = "userID"
ITEM = DEFAULT_ITEM_COL = "itemID"
RATING = DEFAULT_RATING_COL = "rating"
DEFAULT_LABEL_COL = "label"
DEFAULT_TIMESTAMP_COL = "timestamp"
PREDICTION = DEFAULT_PREDICTION_COL = "prediction"
COL_DICT = {
    "col_user": DEFAULT_USER_COL, 
    "col_item": DEFAULT_ITEM_COL, 
    "col_rating": DEFAULT_RATING_COL, 
    "col_prediction": DEFAULT_PREDICTION_COL
}

# Filtering variables
DEFAULT_K = 10
DEFAULT_THRESHOLD = 10

# Random seed
SEED = 74

# Machine epsilon for float32
EPSILON = np.finfo(np.float32).eps

# Text formatting
class T:
   B = '\033[94m' # blue
   G = '\033[92m' # green
   Y = '\033[93m' # yellow
   R = '\033[91m' # red
   b = '\033[1m' # bold
   E = '\033[0m' # end formatting