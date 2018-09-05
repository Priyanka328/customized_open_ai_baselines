import argparse

from .run_mujoco import *

def argsparser_exp():
     parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
     parser.add_argument('--env_id', help='environment ID', default='Hopper')
     parser.add_argument('--seed', help='RNG seed', type=int, default=0)
     
     return parser.parse_args()

if __name__=="__main__":
    args_exp = argsparser_exp()
    args = argsparser()
    args.seed = args_exp.seed
    args.env_id = args_exp.env_id+"-v2"
    args.expert_path='data/stochastic.trpo.'+args_exp.env_id+'.0.00.npz'
    for traj_limitation in [1,5,10,25,50]:
        args.traj_limitation = traj_limitation
        main(args)
