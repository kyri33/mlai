import numpy as np
from sklearn import preprocessing

input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']

encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels) # Creates numbers for each label

print("\nLabel mapping:")
for i, item in enumerate(encoder.classes_): # Comes in the form (i, item)
    print(item, '-->', i)

test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels) # Adds numbers to given labels from previous (encodes list)
print("\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))

encoded_values = [4,2,1]
decoded_labels = encoder.inverse_transform(encoded_values) # Decodes numbers into labels from the encoder
print("Decoded =", list(decoded_labels))
