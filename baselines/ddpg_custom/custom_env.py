"""
    Custom MuJoCo Environment 
"""
import math,sys,os
import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

# Convert quaternion to Euler angle 
def quaternion_to_euler_angle(w, x, y, z):
	ysqr = y * y
	
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + ysqr)
	X = math.degrees(math.atan2(t0, t1))
	
	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	Y = math.degrees(math.asin(t2))
	
	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (ysqr + z * z)
	Z = math.degrees(math.atan2(t3, t4))
	
	return X, Y, Z

""" 
    Ant environment 
""" 
class AntEnvCustom(mujoco_env.MujocoEnv,utils.EzPickle):
    def __init__(self,_CAM_RATE=2.0,_headingCoef=1e-4,_contactCoef=5e-4,_ctrlCoef=1e-1,
                 _VERBOSE=True,_NO_DISPLAY=False):
        # Parse input args
        self.CAM_RATE = _CAM_RATE
        self.headingCoef = _headingCoef # Heading penalty coef for reward
        self.contactCoef = _contactCoef
        self.ctrlCoef = _ctrlCoef
        self.VERBOSE = _VERBOSE
        self.NO_DISPLAY = _NO_DISPLAY
        self.env_name = "ant_custom"
        
        # Load xml
        xmlPath = os.path.dirname(os.path.realpath(__file__))+'/xml/ant_custom.xml'
        # xmlPath = os.getcwd()+'/xml/ant_custom.xml'
        mujoco_env.MujocoEnv.__init__(self, xmlPath, frame_skip=5)
        utils.EzPickle.__init__(self)

        # Some parameters
        D1 = 30
        D2 = 70 
        DX = 30 # 30 / 80
        self.minPosDeg = np.array([-DX,D1,-DX,-D2,-DX,-D2,-DX,D1])
        self.maxPosDeg = np.array([+DX,D2,+DX,-D1,+DX,-D1,+DX,D2])

        # Observation and action dimensions 
        self.obsDim = self.observation_space.shape[0] # 111
        self.actDim = self.action_space.shape[0] # 8

        # Initial positions
        self.initPos = np.array([0,90,0,-90,0,-90,0,90])*np.pi/180.0
        _pos = np.copy(self.init_qpos)
        _pos[7:] = np.array([0,90,0,-90,0,-90,0,90])*np.pi/180.0
        self._initQpos = _pos
        self._initQvel = np.zeros(shape=14)

        # Print out
        if self.VERBOSE:
            print ("Custom Ant Environment.")
            print ("Obs dim:[%d] Act dim:[%d] dt:[%.3f]"%(self.obsDim,self.actDim,self.dt))

        # Do reset once 
        self.reset()

        
    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip) # Run!
        headingAfter = self.get_heading()
        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]
        forward_reward = (xposafter - xposbefore)/self.dt 
        # Modified (upperbound on forward reward)
        if forward_reward > 1.0: 
            forward_reward = 1.0
        # Heading cost
        heading_cost = self.headingCoef*(headingAfter**2+yposafter**2)
        # Control cost
        ctrl_cost = self.ctrlCoef * np.square(a).sum() # 0.1
        # Contact cost
        contact_cost = self.contactCoef * np.sum( # 0.5 * 1e-3
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        # Survive 
        survive_reward = 1.0 
        reward = forward_reward - heading_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done,\
            dict(
                reward_forward=forward_reward,
                reward_heading=-heading_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self,_CAM_RATE=None):
        """
        qpos = self.init_qpos + 0*self.np_random.uniform(size=self.model.nq,low=-.1,high=.1)
        qvel = self.init_qvel + 0*self.np_random.randn(self.model.nv)*.1
        self.set_state(qpos,qvel)
        """
        if _CAM_RATE is not None:
            self.CAM_RATE = _CAM_RATE
            self.viewer_setup()
        self.set_state(self._initQpos,self._initQvel)
        return self._get_obs()

    def set_view(self,_CAM_RATE=None):
        if _CAM_RATE is not None: self.CAM_RATE = _CAM_RATE
        self.viewer_setup()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * self.CAM_RATE

    def get_heading(self):
        q = self.data.get_body_xquat('torso')
        _,_,rZ = quaternion_to_euler_angle(q[0],q[1],q[2],q[3])
        return rZ

    def obs2posDeg(self,_obs):
        return np.asarray(_obs[5:13])*180.0/np.pi 


""" 
    Half Cheetah Environment
""" 
class HalfCheetahEnvCustom(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,_CAM_RATE=2.0,_NO_DISPLAY=False,_VERBOSE=False):

        self.CAM_RATE = _CAM_RATE
        self.NO_DISPLAY = _NO_DISPLAY
        self.VERBOSE = _VERBOSE
        self.env_name = "half_cheetah_custom"

        xmlPath = os.path.dirname(os.path.realpath(__file__))+'/xml/half_cheetah_custom.xml'
        # xmlPath = os.getcwd()+'/xml/half_cheetah_custom.xml'
        mujoco_env.MujocoEnv.__init__(self,xmlPath, frame_skip=5)
        utils.EzPickle.__init__(self)

        # Observation and action dimensions 
        self.obsDim = self.observation_space.shape[0]
        self.actDim = self.action_space.shape[0]

        # Range 
        self.minPosDeg = -20*np.ones(shape=(6))
        self.maxPosDeg = 40*np.ones(shape=(6))

        # Print
        if self.VERBOSE:
            print ("Custom Half Cheetah Environment.")
            print ("Obs dim:[%d] Act dim:[%d] dt:[%.3f]"%
                        (self.obsDim,self.actDim,self.dt))

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum() # 0.1
        reward_run = (xposafter - xposbefore)/self.dt
        if reward_run > 1.0:
            reward_run = 1.0
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],# 
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + 0*self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + 0*self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * self.CAM_RATE

    def get_heading(self):
        q = self.data.get_body_xquat('torso')
        _,_,rZ = quaternion_to_euler_angle(q[0],q[1],q[2],q[3])
        return rZ

    def obs2posDeg(self,_obs):
        return np.asarray(_obs[2:8])*180.0/np.pi 
    
