from keras.models import load_model
import numpy as np


class_names = ['Class1', 'Class2']
total_counter = {'Class1': 0, 'Class2': 0}
right_counter = {'Class1': 0, 'Class2': 0}
model_path = 'trained_models/simpler_CNN.125-0.77.hdf5'
classifier = load_model(model_path, compile=False)

num_classes = 2
input_test = np.load('test_x.npy')
output_test = np.load('test_y.npy')

num = len(output_test)
input_data = np.zeros((1, 7, 4750, 1))
TP = 0
for i in range(num):
    input_data[0] = input_test[i, :, :, :]
    output_data = np.argmax(output_test[i])
    pred = classifier.predict(input_data)
    label_arg = np.argmax(pred)
    if label_arg == output_data:
        TP += 1
        right_counter[class_names[label_arg]] += 1
    total_counter[class_names[output_data]] += 1

print('Total:    Acc = {} '.format(str(TP / num)))

print(' Class1:{}/{}={},\n Class2:{}/{}={}'.format(
    right_counter['Class1'], total_counter['Class1'], (right_counter['Class1'] / total_counter['Class1']),
    right_counter['Class2'], total_counter['Class2'], (right_counter['Class2'] / total_counter['Class2']),
    ))
