import paramiko
import time
import os
import yaml
import concurrent.futures
import socket
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Setup Orca on remote servers")
parser.add_argument("--after-reboot", action="store_true", help="Only run post-reboot commands")
args = parser.parse_args()

# Load configuration from YAML file
CONFIG_FILE = "config.yaml"

with open(CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)

servers = config["servers"]
username = "janechen"

# Local path of the checkpoint file
LOCAL_CHECKPOINT_PATH = "/home/jane/Desktop/Checkpoint-Combined_10RTT_6col_Transformer3_64_5_5_16_4_lr_1e-05-999iter.p"
REMOTE_CHECKPOINT_PATH = "/users/janechen/Orca/models/"

def scp_file(server):
    """ SCP the checkpoint file to the remote server if the branch is 'embedding-input' """
    print(f"Transferring checkpoint file to {server}...")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(server, username=username)
        sftp = client.open_sftp()

        # Ensure the remote directory exists
        try:
            sftp.stat(REMOTE_CHECKPOINT_PATH)  # Check if directory exists
        except FileNotFoundError:
            sftp.mkdir(REMOTE_CHECKPOINT_PATH)  # Create directory if not exists

        # Transfer the file
        sftp.put(LOCAL_CHECKPOINT_PATH, os.path.join(REMOTE_CHECKPOINT_PATH, os.path.basename(LOCAL_CHECKPOINT_PATH)))
        print(f"Checkpoint file successfully transferred to {server}:{REMOTE_CHECKPOINT_PATH}")

        sftp.close()
    except Exception as e:
        print(f"Error transferring checkpoint file to {server}: {e}")
    finally:
        client.close()

def run_remote_commands(server, commands):
    """ SSH into the server and execute the given commands """
    print(f"Connecting to {server}...")
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(server, username=username)
        for cmd in commands:
            print(f"Running on {server}: {cmd}")
            stdin, stdout, stderr = client.exec_command(cmd)
            time.sleep(2)  # Allow time for command execution
            print(stdout.read().decode())
            print(stderr.read().decode())

            if "reboot" in cmd:
                print(f"Reboot command issued for {server}. Disconnecting...")
                client.close()
                return  # Stop execution since the server will go down
    finally:
        client.close()

def wait_for_server(server, timeout=500, interval=10):
    """ Wait until the server is fully booted and SSH is accessible """
    print(f"Waiting for {server} to reboot and become available via SSH...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        # Check if SSH (port 22) is open
        try:
            sock = socket.create_connection((server, 22), timeout=5)
            sock.close()
            print(f"{server} is now fully online with SSH available!")
            return True
        except (socket.timeout, ConnectionRefusedError):
            pass  # SSH is not available yet, keep waiting
        
        time.sleep(interval)
    
    print(f"Timeout: {server} did not come back online within {timeout} seconds.")
    return False


def setup_server(server_config):
    """ Run full setup process for a single server in parallel """
    server = server_config["hostname"]
    branch = server_config["branch"]

    commands_before_reboot = [
        "sudo DEBIAN_FRONTEND=noninteractive apt-get update -y",
        "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential git debhelper autotools-dev dh-autoreconf iptables protobuf-compiler libprotobuf-dev pkg-config libssl-dev dnsmasq-base ssl-cert libxcb-present-dev libcairo2-dev libpango1.0-dev iproute2 apache2-dev apache2-bin gnuplot iproute2 apache2-api-20120211 libwww-perl",
        "sudo chown -R janechen /mydata/",
    
        "cd ~ && git clone https://github.com/Janecjy/ccBench.git",
        "cd ~/ccBench && git submodule update --init --recursive",
        
        "cd ~/ccBench && git clone https://github.com/ravinet/mahimahi",
        "cd ~/ccBench/mahimahi && patch -p1 < ../patches/mahimahi.core.v2.2.patch",
        "cd ~/ccBench/mahimahi && patch -p1 < ../patches/mahimahi.extra.aqm.v1.5.patch",
        "cd ~/ccBench/mahimahi && ./autogen.sh && ./configure && make",
        "cd ~/ccBench/mahimahi && sudo make install",
        "sudo sysctl -w net.ipv4.ip_forward=1",
        
        "mkdir -p ~/venv",
        "sudo apt install -y python3-pip",
        "sudo pip3 install -U virtualenv",
        "virtualenv ~/venv -p python3",
        
        "source ~/venv/bin/activate && pip install --upgrade pip",
        "source ~/venv/bin/activate && pip install gym tensorflow==1.14 sysv_ipc torch",
        
        "cd ~ && git clone https://github.com/soheil-ab/sage.git",
        "cd ~/sage/linux-patch && sudo dpkg -i linux-image-4.19.112-0062_4.19.112-0062-10.00.Custom_amd64.deb",
        "cd ~/sage/linux-patch && sudo dpkg -i linux-headers-4.19.112-0062_4.19.112-0062-10.00.Custom_amd64.deb",
        
        "sudo reboot"
    ]

    commands_after_reboot = [
        "ssh-keyscan -H github.com >> ~/.ssh/known_hosts",

        # Clone Orca if it doesn't already exist
        "[ -d ~/Orca ] || git clone https://github.com/Janecjy/Orca.git ~/Orca",

        # Ensure we're in the Orca directory and checkout the correct branch
        f"cd ~/Orca && git fetch --all && git checkout {branch} && git pull",

        # Apply Git configurations (must be inside a Git directory)
        "cd ~/Orca && git config core.fileMode false",
        "cd ~/Orca && git config --global user.name 'Jane Chen'",
        "cd ~/Orca && git config --global user.email janechen@cs.utexas.edu",

        # Ensure all scripts are executable before running the build
        "cd ~/Orca && chmod +x *.sh",
        "cd ~/Orca && ./build.sh",
        "cd ~/Orca && chmod +x rl-module/mm-thr"
    ]

    # Run commands before reboot
    if not args.after_reboot:
        run_remote_commands(server, commands_before_reboot)

    # Wait for the server to come back online
    if wait_for_server(server):
        # Run commands after reboot
        run_remote_commands(server, commands_after_reboot)

        # If the branch is 'embedding-input', copy the checkpoint file
        if branch == "embedding-input":
            scp_file(server)
    else:
        print(f"Skipping post-reboot setup for {server} due to timeout.")

    print(f"Setup completed for {server}.")

# Run setup on all servers in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=len(servers)) as executor:
    futures = {executor.submit(setup_server, server_config): server_config["hostname"] for server_config in servers}
    
    for future in concurrent.futures.as_completed(futures):
        server = futures[future]
        try:
            future.result()
        except Exception as e:
            print(f"Error setting up {server}: {e}")

print("Setup completed on all servers.")
