import itertools
import flwr as fl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.optimizers import SGD
from typing import Dict, List, Optional, Tuple, Union
import flwr as fl
import numpy as np
#pip install flwr
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import flwr as fl
import tensorflow as tf
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import psutil
from keras.optimizers import SGD
import seaborn as sns
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import json
import subprocess
import os

Total_rounds = 30

# Load configuration parameters from JSON file
def load_config(filename):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config

config = load_config('Attack_config.config')
poison_ratio = -1

def launch_clients():
    # Extract attack parameters from config
    malicious_actor_percentage = config["malicious_actor_percentage"]
    poison_ratio = config["poison_ratio"]
    number_of_malicious_actors = config["number_of_malicious_actors"]
    Attack_Type = config["Attack_Type"]

    print(f"MA% {malicious_actor_percentage} \n PR {poison_ratio} \n NMA {number_of_malicious_actors}")

    # Calculate the number of regular clients and malicious clients
    num_clients = 5
    num_malicious_clients = number_of_malicious_actors
    num_regular_clients = num_clients - num_malicious_clients
    print(f'\n\n num clients {num_regular_clients} num malicious clients: {num_malicious_clients}')

    client_id = 0

    # Launch regular client scripts
    for _ in range(num_regular_clients):
        command = f"bash -c 'python3 client3.py --client_id {client_id} --poison_ratio {poison_ratio}'"
        print(f'EXECUTING REGULAR CLIENT {client_id}')
        subprocess.Popen(command, shell=True)
        client_id += 1
    if(Attack_Type == "labelflip"):
        print("EXECUTING LABELFLIP\n")
        # Launch malicious client scripts
        for _ in range(num_malicious_clients):
            command = f"bash -c 'python3 labelflip_attacker.py --client_id {client_id} --poison_ratio {poison_ratio}'"
            print(f'EXECUTING MALICIOUS CLIENT {client_id}')
            subprocess.Popen(command, shell=True)
            client_id += 1
    else:
        print("NOT LABELFLIPPING")
    if(Attack_Type == "backdoor"):
        print("EXECUTING BACKDOOR\n")
        # Launch malicious client scripts
        for _ in range(num_malicious_clients):
            command = f"bash -c 'python3 Backdoor_attack.py --client_id {client_id} --poison_ratio {poison_ratio}'"
            print(f'EXECUTING MALICIOUS CLIENT {client_id}')
            subprocess.Popen(command, shell=True)
            client_id += 1
    else:
        print("NOT BACKDOOR")
    if (Attack_Type != "backdoor" and Attack_Type != "labelflip"):
        raise Exception("error! attack type not correct, probably a spelling problem in the config file.")

#launch_clients()

#print()

accuracy_history = []
Backdoor_accuracy_history = []
accuracy_hist_name = f"Accuracy Attack_{config['Attack_Type']}_Poisen_{config['poison_ratio']}_MA_{config['malicious_actor_percentage']}_.png"
conf_matrix_name = f"confusion_matrix Attack_{config['Attack_Type']}_Poisen_{config['poison_ratio']}_MA_{config['malicious_actor_percentage']}_.png"
BD_Accuracy = f"backdoor accuracy Attack_{config['Attack_Type']}_Poisen_{config['poison_ratio']}_MA_{config['malicious_actor_percentage']}_.png"

model = tf.keras.applications.MobileNetV2(input_shape=(32, 32, 3), alpha=1.0, include_top=True, weights=None, input_tensor=None, pooling=None, classes=10)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

(x_train_bd, y_train_bd), (x_test_bd, y_test_bd) = tf.keras.datasets.cifar10.load_data()



def implement_backdoor_attack_test():#test for effectiveness of backdoor attack by havinf only backdoor test data
    for index in range(round(len(x_test_bd))-1):
        #assume the pattern of bright pixle and pitch black pixles around is unique
        x_test_bd[index][0, 0] = 255
        x_test_bd[index][0, 1] = 0
        x_test_bd[index][1, 0] = 0
        y_test_bd[index] = 1 

#implement attack for purely testing data        
implement_backdoor_attack_test()


# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    title = title 

    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label, round: ' + str(len(accuracy_history)))

# Plot and save evaluation accuracy over time
def plot_accuracy_history():
    print("Accuracy history is: ", accuracy_history)
    plt.figure(figsize=(8, 6))
    plt.plot(accuracy_history, label="Test Accuracy")
    plt.xlabel("EPOCH (iterations)")
    plt.ylabel("Accuracy (%)")
    plt.title("Federated Learning Accuracy Over Time")
    plt.xlim(0, 30)
    plt.legend()
    plt.grid()

    # Save the plot to a file (e.g., a PNG image)
    plt.savefig(accuracy_hist_name)

    # Plot and save evaluation accuracy over time
def plot_Backdoor_Accuracy():
    print("Backdoor accuracy history is: ", Backdoor_accuracy_history)
    plt.figure(figsize=(8, 6))
    plt.plot(Backdoor_accuracy_history, label="Test Accuracy")
    plt.xlabel("EPOCH (iterations)")
    plt.ylabel("Accuracy (%)")
    plt.title("Federated Learning Accuracy Over Time")
    plt.xlim(0, 30)
    plt.legend()
    plt.grid()

    # Save the plot to a file (e.g., a PNG image)
    plt.savefig(BD_Accuracy)



class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
    self,
    server_round: int,
    results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]]):
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if (server_round == 30):
            # Save results to file after completing the server execution
            save_results_to_file()
            print('results saved.')
        if aggregated_parameters is not None:
            print('########### commencing server evaluation #################')
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            # Set the aggregated parameters to the model
            model.set_weights(aggregated_ndarrays)
            # Compile the model
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            # Perform evaluation on the test dataset
            loss, accuracy = model.evaluate(x_test, y_test)
            print("Evaluation on aggregated model - Loss:", loss, "Accuracy:", accuracy)
            print("\n")
            loss_bd, accuracy_bd = model.evaluate(x_test_bd, y_test_bd)
            print("Evaluation on Backdoored aggregated model - Loss:", loss_bd, "backdoor Accuracy:", accuracy_bd)
            print("\n")
            # Append accuracy to history
            Backdoor_accuracy_history.append(float(accuracy_bd))
            # Append accuracy to history
            accuracy_history.append(float(accuracy))
            print("\n")
            print("Regular accuracy:", accuracy_history)
            print("\n")
            print("\n")
            print("Backdoor accuracy:",Backdoor_accuracy_history, "\n")
            # Generate the accuracy plot
            plot_accuracy_history()
            plot_Backdoor_Accuracy()
            # Generate predictions for the test dataset
            y_pred = np.argmax(model.predict(x_test), axis=1)
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure()
            plot_confusion_matrix(cm, classes=range(10), title=f'Confusion Matrix, Round: {str(len(accuracy_history))}')
            # Save confusion matrix plot
            plt.savefig(conf_matrix_name)
        return aggregated_parameters, aggregated_metrics


    def visualize_similarity_scores(self, similarity_scores: List[float], num_clients: int):
        """Visualize cosine similarity scores."""
        num_pairs = num_clients * (num_clients - 1) // 2  # Calculate the number of pairs
        if len(similarity_scores) != num_pairs:
            print("Error: Incorrect number of similarity scores", num_pairs, similarity_scores)
            return
        print("Similarity scores: ", similarity_scores)
        # Initialize similarity matrix
        similarity_matrix = np.zeros((num_clients, num_clients))

        # Populate similarity matrix with scores
        idx = 0
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                similarity_matrix[i, j] = similarity_scores[idx]
                similarity_matrix[j, i] = similarity_scores[idx]
                idx += 1

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", fmt=".2f", xticklabels=list(range(num_clients)), yticklabels=list(range(num_clients)))
        plt.title("Cosine Similarity Scores Between Clients")
        plt.xlabel("Client ID")
        plt.ylabel("Client ID")
        plt.savefig("cosine_similarity_heatmap.png")


        
    def flatten_parameters(self, parameters):
            """Flatten nested parameters into a single array."""
            if isinstance(parameters, np.ndarray):
                return parameters.flatten()
            elif isinstance(parameters, (list, tuple)):
                flat_parameters = []
                for param in parameters:
                    flat_param = self.flatten_parameters(param)
                    if flat_param is not None:
                        flat_parameters.extend(flat_param)
                    else:
                        print("None value encountered in sub-parameter:", param)
                        return None  # Return None if any sub-parameter is None
                return np.concatenate(flat_parameters)
            elif isinstance(parameters, fl.common.typing.Parameters):
                tensors = parameters.tensors
                tensor_type = parameters.tensor_type
                flat_tensors = []
                for tensor_bytes in tensors:
                    tensor = np.frombuffer(tensor_bytes, dtype=np.float32)
                    flat_tensors.append(tensor)
                return np.concatenate(flat_tensors)
            else:
                print("Unsupported parameter type:", type(parameters))
                return None

    def calculate_similarity_scores(self, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]]) -> List[float]:
        """Calculate cosine similarity scores between model updates."""
        similarity_scores = []
        print("The shape of the results are: ",np.array(results).shape, "\n\n")
        num_clients = len(results)
        for i in range(num_clients):
            for j in range(i + 1, num_clients): 
                result_a = results[i]  # First client
                result_b = results[j]  # Second client
                
                model_update_a = self.flatten_parameters(result_a[1].parameters)
                model_update_b = self.flatten_parameters(result_b[1].parameters)

                dot_product = np.dot(model_update_a, model_update_b)

                magnitude_a = np.linalg.norm(model_update_a)
                magnitude_b = np.linalg.norm(model_update_b)

                # Calculate cosine similarity
                if magnitude_a != 0 and magnitude_b != 0:
                    cosine_similarity_score = dot_product / (magnitude_a * magnitude_b)
                else:
                    cosine_similarity_score = 0.0  # Handle division by zero

                similarity_scores.append(cosine_similarity_score)

        return similarity_scores


def save_results_to_file():
    # Save accuracy history and backdoor accuracy to a JSON file
    results = {
        "\naccuracy_history": accuracy_history,
        "\nAttack_Type: ": config["Attack_Type"],
        "\nBackdoor_accuracy_history": Backdoor_accuracy_history,
        "\npoisen_persentage": config["poison_ratio"],
        "\n% malicious_actors": config["malicious_actor_percentage"],
        "\n Num_Malicious_Actors": config["number_of_malicious_actors"]
    }
     # Define the directory path where files will be saved
    directory = "root/Datapoints"

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
     # Determine the filename
    filename = "results.json"

    # Check if the filename already exists in the directory
    counter = 1
    directory = "Datapoints"
    filename = f"Attack_{config['Attack_Type']}_Poisen_{config['poison_ratio']}_MA_{config['malicious_actor_percentage']}_{counter}.json"

    # Check for unique filename
    while os.path.exists(os.path.join(directory, filename)):
        counter += 1
        filename = f"Attack_{config['Attack_Type']}_Poisen_{config['poison_ratio']}_MA_{config['malicious_actor_percentage']}_{counter}.json"


    # Save results to the JSON file
    with open(os.path.join(directory, filename), 'w') as f:
        json.dump(results, f)


                
# Initialize the FedAvg strategy with the chosen aggregation function
strategy = SaveModelStrategy()

# Start the server with the FedAvg strategy and built-in aggregation
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=Total_rounds),
    strategy=strategy
)


# Save results to file after completing the server execution
#save_results_to_file()
print('results saved.')