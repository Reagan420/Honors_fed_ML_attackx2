import flwr as fl
import tensorflow as tf
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import psutil
from keras.optimizers import SGD
import psutil
import time
import argparse
import json

poisen_persentage = -1
client_id=1
def main():
    print('\n\n\n parsing arguments attacker \n\n\n')
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, help="Client ID")
    parser.add_argument("--poison_ratio", type=float, help="Poison ratio")
    args = parser.parse_args()

    client_id = args.client_id
    poisen_persentage = args.poison_ratio

    print(f"Client ID: {client_id}")
    print(f"Poison Ratio: {poisen_persentage}")

    # Your client script logic goes here
    # Use `client_id` and `poison_ratio` as needed

if __name__ == "__main__":
    main()



# Initialize lists to store evaluation accuracy over time
accuracy_history = []
cpu_percent = psutil.cpu_percent(interval=1)  # Update every 1 second
print("CPU Usage: ", cpu_percent)
# Get memory usage statistics
memory = psutil.virtual_memory()
print(f"Total Memory: {memory.total / (1024 ** 3):.2f} GB")
print(f"Available Memory: {memory.available / (1024 ** 3):.2f} GB")
print(f"Used Memory: {memory.used / (1024 ** 3):.2f} GB")
print(f"Memory Usage Percentage: {memory.percent}%")


model = tf.keras.applications.MobileNetV2(input_shape=(32, 32, 3), alpha=1.0, include_top=True, weights=None, input_tensor=None, pooling=None, classes=10)


if(client_id == -1):
    print('ERROR, client id not set')
    client_id = 1
else:
    print(f"\n\n Poisen Percentage: {poisen_persentage}\n\n")

#not trainable
if(poisen_persentage == -1):
    print('ERROR, poisen persentage id not set')
    poisen_persentage = 1
else:
    print(f"\n\n Poisen Percentage: {poisen_persentage}\n\n")

with open('Attack_config.config', 'r') as f:
        config = json.load(f)

# Extract attack parameters from config
malicious_actor_percentage = config["malicious_actor_percentage"]
poisen_persentage = config["poison_ratio"]
number_of_malicious_actors = config["number_of_malicious_actors"]

print("\n\n CLIENT POISEN %", poisen_persentage)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])#optimizer was adam
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


# Split the training and testing data into 10 arrays
num_clients = 10
x_train_split = np.array_split(x_train, num_clients)
y_train_split = np.array_split(y_train, num_clients)
x_test_split = np.array_split(x_test, num_clients)
y_test_split = np.array_split(y_test, num_clients)

#x_train = x_train_split[client_id]
#y_train = y_train_split[client_id]


def implement_flip_attack():
    # Record start time
    start_time = time.time()
    for index in range(round(len(x_train) * poisen_persentage)):

        # Generate a random label in the specified range
        random_label = np.random.randint(0, 9)
        y_train[index] = random_label

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

implement_flip_attack()


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


