import asyncio
import subprocess
import json

#this script was a attempt too automate the training process but fails due to multithreading issues

def load_config(file):
    with open(file, 'r') as f:
        config = json.load(f)
    return config


def update_config(config, poi_ratio, MA_num, attack_type):
    # Create a new dictionary based on the input configuration
    updated_config = {
        'malicious_actor_percentage': MA_num / 5,
        'poison_ratio': poi_ratio,
        'number_of_malicious_actors': MA_num,
        'Attack_Type': attack_type
    }

    return updated_config


async def launch_server(config):
    # Define the command to run the server script
    command = "python3 server.py"  # Modify this based on your environment

    # Update configuration
    updated_config = update_config(config, *config)

    # Launch the server script asynchronously
    process = await asyncio.create_subprocess_shell(command)
    await process.wait()

    print(f'Server with configuration {config} completed.')


async def launch_servers(configs):
    # Launch servers sequentially, one at a time
    for config in configs:
        await launch_server(config)
        await asyncio.sleep(1)  # Wait for 1 second between launching servers


if __name__ == "__main__":
    config = load_config('Attack_config.config')

    # Define the list of configurations
    configs = [
        (0.7, 1, 'labelflip'),
        (0.7, 2, 'labelflip'),
        (1, 1, 'labelflip'),
        (1, 2, 'labelflip'),
        (0.3, 1, 'labelflip'),
        (0.3, 2, 'labelflip'),
        (0.1, 1, 'labelflip'),
        (0.1, 2, 'labelflip'),
        (0.3, 1, 'backdoor'),
        (0.3, 2, 'backdoor'),
        (0.1, 1, 'backdoor'),
        (0.1, 2, 'backdoor'),
    ]

    # Run the asynchronous function to launch servers sequentially
    asyncio.run(launch_servers(configs))
