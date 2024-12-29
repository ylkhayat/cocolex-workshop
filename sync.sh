#!/bin/bash

ssh thesis "cd /srv/elkhyo/lexquo/generation && tar -czPf basement.tar.gz ./basement"
scp thesis:/srv/elkhyo/lexquo/generation/basement.tar.gz ~/Desktop/masters/local/THESIS/cocolex-dashboard/public/
tar -xzf ~/Desktop/masters/local/THESIS/cocolex-dashboard/public/basement.tar.gz -C ~/Desktop/masters/local/THESIS/cocolex-dashboard/public/
rm ~/Desktop/masters/local/THESIS/cocolex-dashboard/public/basement.tar.gz
node getExperimentsStructure.cjs