#%%
import quimb.tensor as qtn

import pytest

from LiouvilleLanczos.Tensor_networks.MPO_compress import Density_matrix_weighted_compression
from LiouvilleLanczos.Tensor_networks.sweeper import Updater,Sweeper,single_sweep
import numpy as np
#%%
class mock_Updater(Updater):
	def __init__(self):
		self.L = 5
		self.site_log = []
		self._easy_index = np.random.randint(0,self.L)
	def method_name(self):
		return "Mock Updater"
	def update(self,site:int):
		self.site_log.append(site)
		return 1/((len(self.site_log)))**2
	def size(self):
		return self.L
	def easy_index(self):
		return self._easy_index
	def result(self):
		return None

def test_single_sweep():
	for i in range(10):
		updater = mock_Updater()
		cost,direction = single_sweep(updater,1)
		assert direction == 1 or direction == -1, "direction is either 1 or -1"
		assert updater.easy_index() == updater.site_log[0]+1 or updater.easy_index() == updater.site_log[0]-1 ,"start at a site neighboring easy_index"
		assert updater.easy_index() == updater.site_log[-1], "finishes at the easy index"
		assert np.all(np.abs(np.diff(updater.site_log))==1), "single site steps only"

def test_sweeper():
	updater = mock_Updater()
	out = Sweeper(updater,1/100)
	sl = np.array(updater.site_log)
	assert np.all(sl>=0), "went out of bounds"
	assert np.all(sl<updater.size()), "went out of bounds"
	assert np.all(np.abs(np.diff(updater.site_log))==1), "single site steps only"
# %%
def test_DMWU():
	input_MPOs = [qtn.MPO_rand_herm(5,2,4) for i in range(2)]
	state = qtn.MPS_rand_state(5,5,4)
	print(state.site_ind_id)
	print(input_MPOs[0].upper_ind_id,' ',input_MPOs[0].lower_ind_id)
	statec = state.conj()
	statec.site_ind_id = input_MPOs[0].lower_ind_id
	rho = state &statec
	print(rho[0])
# %%
test_DMWU()
# %%
