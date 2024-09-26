#!/bin/bash

# Define the base IP address
BASE_IP="192.168.42."

# Loop through the range from 10 to 109, excluding 12
for i in {108..108}; do
    TARGET_IP="${BASE_IP}${i}"

    echo "Connecting to ${TARGET_IP}..."

    ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -A -i /home/lorenzo/Scrivania/lavoro/ssh.key -J user@130.251.7.71 root@"${TARGET_IP}" << 'ENDSSH'
        echo "Updating system packages..."
        sudo apt update

        echo "Installing Docker..."
        sudo apt install docker.io -y

        echo "Pulling Docker image..."
        docker pull docker.io/lorenzofoschi/stonkspp_debug:latest

        echo "Running Docker container..."
        docker run -p 2222:22 --privileged --cgroupns=host --name stonkss --security-opt apparmor=unconfined lorenzofoschi/stonkspp_debug:latest &

        echo "Executing command inside the container..."
        sleep 20
        docker exec -d stonkss /bin/bash -c "cd ../cosima_core/mosaik_sim/ && sudo nohup python3 runs.py > nohup.out 2>&1 &"
ENDSSH

    echo "Commands executed on ${TARGET_IP}."
done
