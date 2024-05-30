import flwr as fl
import tensorflow as tf
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import psutil
import time
import os
import json


#normal back door (on pixle top right)
#krumb 


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


client_id = 2

model = tf.keras.applications.MobileNetV2(input_shape=(32, 32, 3), alpha=1.0, include_top=True, weights=None, input_tensor=None, pooling=None, classes=10)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Split the training and testing data into 10 arrays
num_clients = 10
x_train_split = np.array_split(x_train, num_clients)
y_train_split = np.array_split(y_train, num_clients)
x_test_split = np.array_split(x_test, num_clients)
y_test_split = np.array_split(y_test, num_clients)




def save_image_before_and_after(index, output_dir):
    # Original image
    original_image = x_train[index]
    original_label = y_train[index]

    # Apply backdoor pattern to create modified image
    modified_image = np.copy(original_image)
    modified_image[0, 0] = 255  # Bright pixel
    modified_image[0, 1] = 0     # Pitch black pixel
    modified_image[1, 0] = 0     # Pitch black pixel

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save images
    original_filename = os.path.join(output_dir, f'original_image_{index}.png')
    modified_filename = os.path.join(output_dir, f'modified_image_{index}.png')

    plt.imsave(original_filename, original_image)
    plt.imsave(modified_filename, modified_image)

    print(f"Original Image (Class: {original_label}): {original_filename}")
    print(f"Modified Image (Backdoor Attack): {modified_filename}")

# Example: Save before and after images for a specific index (e.g., 0)
index_to_save = 2
output_directory = 'image_outputs'
save_image_before_and_after(index_to_save, output_directory)


with open('Attack_config.config', 'r') as f:
        config = json.load(f)

# Extract attack parameters from config
malicious_actor_percentage = config["malicious_actor_percentage"]
poisen_persentage = config["poison_ratio"]
number_of_malicious_actors = config["number_of_malicious_actors"]

print("\n\n CLIENT POISEN %", poisen_persentage)

x_train = x_train_split[client_id]
y_train = y_train_split[client_id]

def implement_backdoor_attack():
    # Record start time
    start_time = time.time()
    for index in range(round(len(x_train) * poisen_persentage)):
        #assume the pattern of bright pixle and pitch black pixles around is unique
        x_train[index][0, 0] = 255
        x_train[index][0, 1] = 0
        x_train[index][1, 0] = 0
        y_train[index] = 1 
        #print(index)
        if index == (len(x_train) * poisen_persentage)-1:
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

    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")

implement_backdoor_attack()


class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train_split[client_id], y_train_split[client_id], epochs=1, batch_size=32)
        return model.get_weights(), len(x_train_split[client_id]), {}
    
''' #old code
accuracy_history = []
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}
  '''  

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

# Generate the accuracy plot
plot_accuracy_history()
