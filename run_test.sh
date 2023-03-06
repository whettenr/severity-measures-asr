#!/usr/bin/env bash
python EX_3_run.py ctcbycos && python EX_3_run.py base && python EX_3_run_last5.py ctcbycos && python EX_3_run.py cos


# currently running
# nohup python EX_3_run.py base &

# need to run with 512 rnnu
# nohup python EX_3_run.py ctcbycos &
# nohup python EX_3_run_last5.py ctcbycos &

# nohup python EX_3_run_last5.py ctcbycos && python EX_3_run.py ctcbycos && python EX_3_run.py base && python EX_3_run.py cos &
# nohup python EX_3_run_last5.py ctcbycos && python EX_3_run.py base && python EX_3_run.py ctcbycos &
# nohup python EX_3_run.py base &&  nohup python EX_3_run.py ctcbycos && nohup python EX_3_run_last5.py &
