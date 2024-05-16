# This repo is under dev
Note: graphs and mosaik_sim should be placed in the cosima dir, to then build with the Dockerfile:

docker run -v <path>/cosima/cosima_core/mosaik_sim:/root/models/cosima_core/mosaik_sim -p 2222:22 --privileged --cgroupns=host sim
