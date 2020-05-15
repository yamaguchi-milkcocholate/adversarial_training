import pickle
from run_gtsrb_tf import gen_extra_data, pre_process_image
import numpy as np
from sklearn.model_selection import train_test_split


training_file = 'src/storage/datasets/GTSRB/train.p'
testing_file = 'src/storage/datasets/GTSRB/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

X_train_SS, X_valid_SS, y_train_SS, y_valid_SS = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    random_state=22
)

ang_rot = 10 * 0.9 ** 1
trans_rot = 2 * 0.9 ** 1
shear_rot = 2 * 0.9 ** 1

Image_train_GS_rot, y_train_rot, labels_train_rot = gen_extra_data(
    X_train_SS,
    y_train_SS,
    43,
    5,
    ang_rot,
    trans_rot,
    shear_rot,
    1
)
with open('src/storage/datasets/GTSRB/tf_train.p', 'wb') as f:
    pickle.dump({
        'features': Image_train_GS_rot,
        'labels': y_train_rot
    }, f)

image_GS_valid = np.array(
    [pre_process_image(X_valid_SS[i]) for i in range(len(X_valid_SS))], dtype=np.float32)
# labels_valid_SS = OHE_labels(y_valid_SS, 43)
with open('src/storage/datasets/GTSRB/tf_valid.p', 'wb') as f:
    pickle.dump({
        'features': image_GS_valid,
        'labels': y_valid_SS
    }, f)

image_GS_test = np.array([pre_process_image(X_test[i]) for i in range(len(X_test))], dtype=np.float32)
with open('src/storage/datasets/GTSRB/tf_test.p', 'wb') as f:
    pickle.dump({
        'features': image_GS_test,
        'labels': y_test
    }, f)
