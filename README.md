# This repo is under dev
Note: graphs and mosaik_sim should be placed in the cosima dir, to then build with the Dockerfile:

docker run -v <path>/cosima/cosima_core/mosaik_sim:/root/models/cosima_core/mosaik_sim -p 2222:22 --privileged --cgroupns=host sim

To then being able to use Gurobi you have to copy your gurobi.lic in the home dir: 
cp gurobi.lic ~

It is then suggested to create an alias for closing the 4242 port (cosima leaves it open if the program is not properly closed):
sudo apt install lsof &
alias xxx='sudo kill -9 $(sudo lsof -t -i :4242)'

This is the detailed report: https://docs.google.com/document/d/1dhX8BoJopfouI9Ja6J47jQ3_RNiKPb7_Iijr7ihu9wE/edit?usp=sharing
