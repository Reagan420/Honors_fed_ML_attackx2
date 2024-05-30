import flwr as fl
import tensorflow as tf
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import psutil
import time


#this attack was never fully implemented


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


def Data_Priming_attack():
    # Record start time
    start_time = time.time()
    
    # Opacity values for original and additional images
    original_opacity = 0.8
    additional_opacity = 0.2
    
    for index in range(len(x_train)):
        # Create additional image with random content
        additional_image = np.random.randint(0, 256, size=(32, 32, 3)).astype(np.uint8)
        
        # Combine original and additional images with specified opacities
        blended_image = (original_opacity * x_train[index] + additional_opacity * additional_image).astype(np.uint8)
        
        # Update the training data with the blended image
        x_train[index] = blended_image
        
        # Generate a random label in the specified range
        random_label = np.random.randint(0, 9)
        y_train[index] = random_label

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

    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")

# Call the function to implement the opacity attack
Data_Priming_attack()


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

# Generate the accuracy plot
plot_accuracy_history()
