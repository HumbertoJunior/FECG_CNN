import numpy as np
import matplotlib.pyplot as plt2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten


#Preparing the samples 60 by 60
def seq2samples(seq, sample_len=60):
  samples = []
  for i in range(len(seq) - sample_len):
    samples.append(seq[i : i + sample_len])
  return samples

# Naming the samples
def label_data(labels, samples, delta=0, skew=0):
  sample_len = len(samples)
  label_array = np.zeros((sample_len,), dtype=np.int)
  for label in labels:
    true_sample = label[0] - skew
    label_array[true_sample] = 1
    for i in range(1, delta):
      if true_sample + i < sample_len:
        label_array[true_sample + i] = 1
      if true_sample - i > 0:
        label_array[true_sample - i] = 1
  return label_array

#
def one_hot(labels):
  b = np.zeros((labels.size, labels.max()+1))
  b[np.arange(labels.size), labels] = 1
  return b

# Mixing Samples
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def seq2labels(label_seq, aecg_seq, offset=-20, size=60):
#   for i, l in enumerate(label_seq):
#     if l == 1 and aecg_seq[i] > 0.4:
#       label_seq[i] = 0
  return np.roll(label_seq, offset)[:-size]


    #READING FILE
import csv

abdomen_crop_seq = []
label_crop_seq = []

with open('/Users/HumbertoJunior/Documents/MATLAB/final.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        abdomen_crop_seq.append(float(row['aecg']))
        label_crop_seq.append(int(row['label']))
    
print(abdomen_crop_seq[:10], label_crop_seq[:10])



#Dividing Samples using seq2samples
abdomen_samples = np.asarray(seq2samples(abdomen_crop_seq))

print(abdomen_samples.shape)

# Labeling data
labels = []

labels = seq2labels(label_crop_seq, abdomen_crop_seq)

# Cheking samples
for i in range(0, len(labels)):
  if labels[i] == 1:
    plt2.plot(abdomen_samples[i])
    plt2.title('Amostra contendo pico FECG')
    plt2.xlabel('Nº da amostra')
    plt2.ylabel('Amplitude[mV]')
    plt2.show()
    break

print(abdomen_samples.shape, labels.shape)
print(labels[:136])

x = abdomen_samples
y = labels

x, y = unison_shuffled_copies(x, y)

print(x[:1],x.shape,y[:1],y.shape)

indexes = []

num_neg_samples = 3214
neg_samples_count = 0
for i, d in enumerate(y):
  if d == 1:
    indexes.append(i)
  elif num_neg_samples  > neg_samples_count:
    indexes.append(i)
    neg_samples_count += 1
    
x, y = unison_shuffled_copies(x[indexes], y[indexes])

print(x[:1],x.shape,y[:1],y.shape)

# Splitting data 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(np.expand_dims(x, axis=2), y, test_size=0.4, random_state=0)
print(X_train.shape,X_test.shape)

# Create the model
model = Sequential() # linear stack of neural network layers 
model.add(Conv1D(filters=32, kernel_size=7, padding='same', activation='relu', input_shape=(60, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Conv1D(filters=32, kernel_size=1, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
print(model.summary())
model.fit(X_train, Y_train, epochs=20, batch_size=128, verbose=1)
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

# # Printing results
result = model.predict(X_test)

print(result[:50])

count = 0

for i, r in enumerate(result):
    plt2.plot(X_test[i])
    plt2.xlabel('Nº da amostra')
    plt2.ylabel('Amplitude[mV]')
    plt2.title('Classificação real: ' + str(Y_test[i]) + '        ' + 'Classificação da rede: '+ str(r))
    plt2.show()
    print(count)
    count += 1
    if count == 50:
        break


plt2.plot(X_test[39])
print('{', end='')
for i in range (0,len(X_test[21])):
    print('{0:.32f}'.format(X_test[21][i][0]), end = '')
    print(',', end = '')
print('};')


count = 0

for i, r in enumerate(result):
    if r > 0.5 and Y_test[i] == 0:
        plt2.plot(X_test[i])
        plt2.xlabel('Nº da amostra')
        plt2.ylabel('Amplitude[mV]')
        plt2.title('Classificação real: ' + str(Y_test[i]) + '        ' + 'Classificação da rede: '+ str(r))
        plt2.show()
        count += 1
    if count == 10:
        break
        

# # # Printing results
# result = model.predict(X_test)

# print(result[:4])

# count = 0
# count2 = 1
# j = 1 
# plt2.figure(1)
# for i, r in enumerate(result):
#     if r > 0.5:
#         if count2 > 4:
#             j+=1
#             count2 = 1
            
#         plt2.subplot(8,j,count2)
#         plt2.plot(X_test[i])
#         plt2.xlabel('Nº da amostra')
#         plt2.ylabel('Amplitude[mV]')
#     #     plt2.title('Conjunto '+ str(i))
#         plt2.show()
#         print('Classificação real: ' + str(Y_test[i]))
#         print('Classificação da rede: '+ str(r))
#         count += 1
#         count2 += 1
#     if count == 8:
#         break



 # Attempt to convert to C code
model.save('tcc2_model_export.h5', include_optimizer=False)


import keras.backend as K


def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def display_activations(activation_maps):
    import numpy as np
    import matplotlib.pyplot as plt
    """
    (1, 26, 26, 32)
    (1, 24, 24, 64)
    (1, 12, 12, 64)
    (1, 12, 12, 64)
    (1, 9216)
    (1, 128)
    (1, 128)
    (1, 10)
    """
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        if len(shape) == 4:
            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')
        plt.imshow(activations, interpolation='None', cmap='jet')
        plt.show()


a = get_activations(model,X_test)


conjunto = 39


transp_a_samples = a[1][39]
transp_a_samples = transp_a_samples.transpose()
print(transp_a_samples.shape)
plt2.plot(transp_a_samples[0])


# print(a[2][21][29][31])
#a[layer][conjunto][amostras][filtros]


for i in range (0,len(a)):
    plt2.figure(i)
    if i > 5:
        plt2.plot(a[i][conjunto],marker = 'o', ls = ':')
    else:
        plt2.plot(a[i][conjunto])
    
    if i > 5 and i < 8 :
        plt2.xlabel('Nº do neurônio')
        plt2.ylabel('Importância')
    
    elif i == 8:
        plt2.xlabel('Nº do neurônio')
        plt2.ylabel('Porcentagem de acerto [%]')
        print(a[i][conjunto])
       
    else:
        plt2.xlabel('Nº da amostra')
        plt2.ylabel('Amplitude[mV]')
   
    plt2.title('Camada '+ str(i+1))
    plt2.show()


W_1_layer = model.layers[0].get_weights()[0]


np.set_printoptions(threshold=1000,precision = 32)
t_W_1_layer = np.reshape(W_1_layer,(7,32))
vwc1 = t_W_1_layer.transpose()
print('{')
for j in range (0,32):
    print('{',end='')
    for i in range (0,7):
        print('{:.32f}'.format(vwc1[j][i]), end='') 
        if(i != 6):
            print(',',end='')
    print('},')
print('}')


B_1_layer = model.layers[0].get_weights()[1]


np.set_printoptions(threshold=np.nan)
B_1_layer = model.layers[0].get_weights()[1]
print('{')
for i in range (0,len(B_1_layer)):
    print('{:.32f}'.format(B_1_layer[i]), end='') 
    if(i != (len(B_1_layer)-1)):
        print(',',end='')
print('}')


W_2_layer = model.layers[2].get_weights()[0]
np.set_printoptions(threshold=np.nan)
cw2 = W_2_layer.transpose()

print('{', end = '')
for k in range (0,32):
    print('{', end = '')
    for j in range (0,32):
        print('{',end='')
        for i in range (0,5):
            print('{:.32f}'.format(cw2[k][j][i]), end='') 
            if(i != 4):
                print(',',end='')
        print('}',end='')
        if(j != 31):
            print(',')
    print('}',end='')
    if(k != 31):
        print(',')
print('}')


B_2_layer = model.layers[2].get_weights()[1]
print('{')
for i in range (0,len(B_2_layer)):
    print('{:.32f}'.format(B_2_layer[i]), end='') 
    if(i != (len(B_2_layer)-1)):
        print(',',end='')
print('}')


W_3_layer = model.layers[3].get_weights()[0]
np.set_printoptions(threshold=np.nan)
cw3 = W_3_layer.transpose()

print('{')
for k in range (0,32):
    print('{')
    for j in range (0,32):
        print('{',end='')
        for i in range (0,3):
            print('{:.32f}'.format(cw3[k][j][i]), end='') 
            if(i != 4):
                print(',',end='')
        print('}',end='')
        if(j != 31):
            print(',')
    print('}',end='')
    if(k != 31):
        print(',')
print('}')


B_3_layer = model.layers[3].get_weights()[1]
print('{')
for i in range (0,len(B_3_layer)):
    print('{:.32f}'.format(B_3_layer[i]), end='') 
    if(i != (len(B_3_layer)-1)):
        print(',',end='')
print('}')

W_4_layer = model.layers[4].get_weights()[0]
np.set_printoptions(threshold=np.nan)
cw4 = W_4_layer.transpose()

print('{')
for k in range (0,32):
    print('{')
    for j in range (0,32):
        print('{',end='')
        for i in range (0,1):
            print('{:.32f}'.format(cw4[k][j][i]), end='') 
            if(i != 4):
                print(',',end='')
        print('}',end='')
        if(j != 31):
            print(',')
    print('}',end='')
    if(k != 31):
        print(',')
print('}')

B_4_layer = model.layers[4].get_weights()[1]
print('{')
for i in range (0,len(B_4_layer)):
    print('{:.32f}'.format(B_4_layer[i]), end='') 
    if(i != (len(B_4_layer)-1)):
        print(',',end='')
print('}')





W_1_fully_layer = model.layers[6].get_weights()[0]
np.set_printoptions(threshold=np.nan)
fw1 = W_1_fully_layer.transpose()

print('{')
for j in range (0,128):
    for i in range (0,960):
        print(fw1[j][i], end='') 
        print(',',end='')
print('}')



# print('{')
# for j in range (0,128):
#     print('{')
#     for i in range (0,960):
#         print(fw1[j][i], end='') 
#         if(i != (len(fw1)-1)):
#             print(',',end='')
#     print('}',end='')
#     if(j != 127):
#         print(',')
# print('}')



B_1_fully_layer = model.layers[6].get_weights()[1]
print('{')
for i in range (0,len(B_1_fully_layer)):
    print(B_1_fully_layer[i], end='') 
    if(i != (len(B_1_fully_layer)-1)):
        print(',',end='')
print('}')


W_2_fully_layer = model.layers[7].get_weights()[0]
np.set_printoptions(threshold=np.nan)
fw2 = W_2_fully_layer.transpose()

print('{')
for j in range (0,64):
    for i in range (0,128):
        print(fw2[j][i], end='') 
        print(',',end='')
print('}')



B_2_fully_layer = model.layers[7].get_weights()[1]
print('{')
for i in range (0,len(B_2_fully_layer)):
    print(B_2_fully_layer[i], end='') 
    if(i != (len(B_2_fully_layer)-1)):
        print(',',end='')
print('}')


W_3_fully_layer = model.layers[8].get_weights()[0]
np.set_printoptions(threshold=np.nan)
fw3 = W_3_fully_layer.transpose()

print('{')
for j in range (0,1):
    print('{')
    for i in range (0,64):
        print(fw3[j][i], end='') 
        if(i != (len(fw3)-1)):
            print(',',end='')
    print('}') 
print('}')



B_3_fully_layer = model.layers[8].get_weights()[1]
print('{')
for i in range (0,len(B_3_fully_layer)):
    print(B_3_fully_layer[i], end='') 
    if(i != (len(B_3_fully_layer)-1)):
        print(',',end='')
print('}')