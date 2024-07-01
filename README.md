# This repo is under dev

To use the full simulator within omnetpp (so with cosima) you have to clone cosima (see cosima repo) and place "graphs" and "mosaik_sim" directories in the "cosima_core" directory, to then build with the slightly modified Dockerfile provided in this repository using: docker build -t <x>:latest .

Then, once the container of cosima is built (with the modified Dockerfile and with inside our two directories), it's time to run it (mounting mosaik_sim directory to avoid losing changes in the code, or even mounting all "models" directory if also changes in the network are required): 

docker run -v <path>/cosima/cosima_core/mosaik_sim:/root/models/cosima_core/mosaik_sim -p 2222:22 --privileged --cgroupns=host sim

To then being able to use Gurobi you have to copy your gurobi.lic in the home dir: 
cp gurobi.lic ~

It is then suggested to create an alias for closing the 4242 port (cosima leaves it open if the program is not properly closed):
sudo apt install lsof &
alias x='sudo kill -9 $(sudo lsof -t -i :4242)'

This is the detailed report: https://docs.google.com/document/d/1dhX8BoJopfouI9Ja6J47jQ3_RNiKPb7_Iijr7ihu9wE/edit?usp=sharing
