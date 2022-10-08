import pickle
import wandb
import argparse

parser = argparse.ArgumentParser(description="load pickle file")

parser.add_argument("--save-path",type = str,nargs = '?',
                    required=True,help="Input file path ")
parser.add_argument("--wandb",action = "store_false",help = "Use wandb or not")
parser.add_argument("--alg", type=str, nargs="?",
                    default="maddpg", help="Please enter the alg name.")
parser.add_argument("--scenario", type=str, nargs="?", default="case33_3min_final",
                    help="Please input the valid name of an environment scenario.")
parser.add_argument("--env", type=str, nargs="?",
                    default="var_voltage_control", help="Please enter the env name.")
parser.add_argument("--test-mode", type=str, nargs="?", default="test_data",
                    help="Please input the valid test mode: single or batch.")
parser.add_argument("--alias", type=str, nargs="?", default="",
                    help="Please enter the alias for exp control.")
parser.add_argument("--mode", type=str, nargs="?", default="distributed",
                    help="Please enter the mode: distributed or decentralised.")
parser.add_argument("--voltage-barrier-type", type=str, nargs="?", default="l1",
                    help="Please input the valid voltage barrier type: l1, courant_beltrami, l2, bowl or bump.")

argv = parser.parse_args()

net_topology = argv.scenario

log_name = "-".join([argv.env, net_topology, argv.mode,
                    argv.alg, argv.voltage_barrier_type, argv.alias])

# If you want to use wandb, please replace project and entity with yours.
if argv.wandb:
    wandb.init(
        project='test_T-MAAC',
        name=log_name,
        group=log_name.split('_')[2] +'-'+ log_name.split('_')[-1],
        save_code=True
    )
    wandb.run.log_code('.')

LOAD_PATH = argv.save_path + '/test_record_'+log_name+'_'+argv.test_mode+'.pickle'

with open(LOAD_PATH,"rb") as f:
    data = pickle.load(f)

for month in range(1,13):
    for k,v in data[month].items():
        wandb.log({k+"_mean":data[month][k][0],
                    k+"_std":data[month][k][1]},month)

print(data)
