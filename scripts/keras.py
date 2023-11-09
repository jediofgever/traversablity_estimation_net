#!/usr/bin/env python3            

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, ReLU, Concatenate
import visualkeras

def create_pointnet_curv_model():
    # Define the inputs
    point_input = Input(shape=(None, 6))  # Assuming the point cloud input is of shape [batch_size, num_points, 6]
    imu_input = Input(shape=(13,))  # IMU input with 13 features

    # Conv1D layers with ReLU activations
    x = Conv1D(128, 1, activation='relu')(point_input)
    x = Conv1D(256, 1, activation='relu')(x)
    
    # Global Max Pooling layer
    x = GlobalMaxPooling1D()(x)
    
    # Dense layers with ReLU activations
    x = Dense(512, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    
    # Concatenate the additional IMU data
    concatenated = Concatenate()([x, imu_input])
    
    # Final Dense layer for output
    output = Dense(1)(concatenated)
    
    # Create the model
    model = Model(inputs=[point_input, imu_input], outputs=output)
    return model

# Create the Keras PointnetCurv model
model = create_pointnet_curv_model()

# You can print the model summary to verify its architecture
model.summary()

visualkeras.layered_view(model).show() # display using your system viewer
visualkeras.layered_view(model, to_file='output.png') # write to disk
visualkeras.layered_view(model, to_file='output.png').show() # write and show