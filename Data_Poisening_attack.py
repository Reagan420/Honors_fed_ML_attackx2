import flwr as fl
import tensorflow as tf
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import psutil
import time

#python3 clientLabelflippingAttack.py
###########################THis attack was discontinued

# Initialize lists to store evaluation accuracy over time
accuracy_history = []
print("#############################################################################")
cpu_percent = psutil.cpu_percent(interval=1)  # Update every 1 second
print(f"CPU Usage: {cpu_percent:.9f}%")
# Get memory usage statistics
memory = psutil.virtual_memory()
print(f"Total Memory: {memory.total / (1024 ** 3):.9f} GB")
print(f"Available Memory: {memory.available / (1024 ** 3):.9f} GB")
print(f"Used Memory: {memory.used / (1024 ** 3):.9f} GB")
print(f"Memory Usage Percentage: {memory.percent}%")
print("#############################################################################")


model = tf.keras.applications.MobileNetV2(input_shape=(32, 32, 3), alpha=1.0, include_top=True, weights=None, input_tensor=None, pooling=None, classes=10)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()



def implement_data_poisoning_attack():
    # Record start time
    start_time = time.time()
    
    # Percentage of data to poison
    poison_percentage = 0.1  # Adjust this value as needed
    
    # Number of data points to poison
    num_poisoned_points = int(len(x_train) * poison_percentage)
    
    # Randomly select indices for poisoning
    poison_indices = np.random.choice(len(x_train), num_poisoned_points, replace=False)
    
    # Iterate through the selected indices and add noise to each pixel
    for index in poison_indices:
        # Add poison pattern to the image by adding noise to each pixel (max 25% noise per pixel)
        noise = np.random.normal(loc=0, scale=0.25*np.max(x_train[index]), size=x_train[index].shape)  # Gaussian additive noise
        
        # Convert the noise array to the same data type as x_train[index]
        noise = noise.astype(x_train[index].dtype)
        
        # Apply random noise to each pixel in the image
        x_train[index] += noise
        
        # Clip pixel values to be within the valid range [0, 255]
        x_train[index] = np.clip(x_train[index], 0, 255)

        # Print the noise for debugging
        if index == 1:
            print(f"Noise added to image at index {index}:")
            print(noise)
        
        # Check CPU and memory usage halfway through the loop
        if index == len(x_train) // 2:
            cpu_percent = psutil.cpu_percent(interval=1)  # Update every 1 second
            print("=============================================================")
            print(f"CPU Usage: {cpu_percent:.9f}%")
            # Get memory usage statistics
            memory = psutil.virtual_memory()
            print(f"Total Memory: {memory.total / (1024 ** 3):.9f} GB")
            print(f"Available Memory: {memory.available / (1024 ** 3):.9f} GB")
            print(f"Used Memory: {memory.used / (1024 ** 3):.9f} GB")
            print(f"Memory Usage Percentage: {memory.percent}%")
            print("=============================================================")

    # Record end time
    end_time = time.time()
    
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    # Print the elapsed time
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")


# Call the function to implement the data poisoning attack
implement_data_poisoning_attack()


class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        # Append accuracy to history
        accuracy_history.append(float(accuracy))
        return loss, len(x_test), {"accuracy": float(accuracy)}

# Start the client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CifarClient())

# Plot and save evaluation accuracy over time
def plot_accuracy_history():
    print("Accuracy history is: ",accuracy_history)
    plt.figure(figsize=(8, 6))
    plt.plot(accuracy_history, label="Test Accuracy")
    plt.xlabel("Time (iterations)")
    plt.ylabel("Accuracy")
    plt.title("Evaluation Accuracy Over Time")
    plt.legend()
    plt.grid()

    # Save the plot to a file (e.g., a PNG image)
    plt.savefig("accuracy_over_time.png")

