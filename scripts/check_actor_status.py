import paramiko
import re
import json
from concurrent.futures import ThreadPoolExecutor

# Load config from JSON
with open("actor.json") as f:
    config = json.load(f)

actor_nodes = config["actor_nodes"]
USERNAME = config["username"]
DOMAIN_SUFFIX = config["domain_suffix"]
LOG_DIR = config["log_dir"]
SSH_KEY = config["ssh_key"]  # can be None

def parse_log_tail(log_content):
    """Parse tail output and return latest epoch and reward if found."""
    lines = log_content.strip().splitlines()
    for line in reversed(lines):
        if "Epoch" in line and "OverallReward" in line:
            match = re.search(r'"Epoch":\s*(\d+).*?"OverallReward":\s*([0-9.]+)', line)
            if match:
                return match.group(1), match.group(2)
    return None, None

def fetch_node_status(hostname):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    results = []

    try:
        ssh.connect(
            hostname=f"{hostname}{DOMAIN_SUFFIX}",
            username=USERNAME,
            key_filename=SSH_KEY
        )

        # Find all actor log files
        find_cmd = f"cd {LOG_DIR} && ls *-actor*.txt"
        stdin, stdout, stderr = ssh.exec_command(find_cmd)
        log_files = stdout.read().decode().splitlines()

        for log_file in log_files:
            actor_id_match = re.search(r'-actor(\d+)\.txt$', log_file)
            if not actor_id_match:
                continue
            actor_id = int(actor_id_match.group(1))

            tail_cmd = f"tail -n 20 {LOG_DIR}/{log_file}"
            stdin, stdout, stderr = ssh.exec_command(tail_cmd)
            content = stdout.read().decode()

            epoch, reward = parse_log_tail(content)
            if epoch and reward:
                results.append((actor_id, epoch, reward))
            else:
                results.append((actor_id, "N/A", "N/A"))
    except Exception as e:
        results.append(("ERROR", hostname, str(e)))
    finally:
        ssh.close()

    return hostname, results

def main():
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_node_status, hostname) for hostname in actor_nodes.values()]
        for future in futures:
            hostname, actor_results = future.result()
            print(f"--- {hostname} ---")
            for result in sorted(actor_results):
                if result[0] == "ERROR":
                    print(f"[ERROR] {result[1]}: {result[2]}")
                else:
                    print(f"Actor {result[0]:>3}: Epoch {result[1]:>6}, Reward {result[2]}")
            print()

if __name__ == "__main__":
    main()
