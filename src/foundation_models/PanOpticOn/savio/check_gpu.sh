sinfo |  tr -s ' ' > sinfo.csv
squeue |  tr -s ' ' > squeue.csv
python savio/savio_comp_usage.py