#!/bin/bash

ssh thesis "tar -czPf /srv/elkhyo/lexquo/generation/basement.tar.gz /srv/elkhyo/lexquo/generation/basement"
scp thesis:/srv/elkhyo/lexquo/generation/basement.tar.gz ~/Desktop/masters/local/THESIS/cocolex-dashboard/
tar -xzf ~/Desktop/masters/local/THESIS/cocolex-dashboard/basement.tar.gz -C ~/Desktop/masters/local/THESIS/cocolex-dashboard/
rm ~/Desktop/masters/local/THESIS/cocolex-dashboard/basement.tar.gz
node getExperimentsStructure.cjs