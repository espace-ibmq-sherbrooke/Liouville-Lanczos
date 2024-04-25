
from typing import Optional
from abc import abstractmethod,ABC


class Updater(ABC):
	"""
	abstract base class of updater objects for the abstract sweeper function.
	"""
	def method_name(self):
		return "abstract updater"
	@abstractmethod
	def update(self,site:int):
		...
	@abstractmethod
	def size(self):
		...
	@abstractmethod
	def easy_index(self):
		...
	@abstractmethod
	def result(self):
		...

def single_sweep(updater:Updater,direction:int):
	"""
	sweep the line tensor network from starting at a neighbor of the easy site and finishing at the easy site.
	"""
	L = updater.size()
	oc = updater.easy_index()
	if oc >= L - 1:
		direction = -1
		oc = L - 1
	count = 0
	while count < 2*(L-1):
		oc += direction
		out = updater.update(oc)
		direction = direction*(1 - 2 * (oc == (L - 1) or oc == 0)) #flip the direction when we reach the edge
		count += 1
	return out,direction

def relative_convergence(new_cost,old_cost):
	"""
	detect the effective exponential convergence ratio.
	This number goes down when convergence is slower than an exponential.
	"""
	return abs(2 * (new_cost - old_cost) / (new_cost + old_cost))

def Sweeper(updater:Updater,crit:float,max_sweep:int=1000,convergence_eval=relative_convergence):
	"""
	Abstract sweeper, aims be suitable for any few-sites update strategy for a linear tensor network output (e.g. DMRG).
	
	updater: implement the specifics of the update strategy
	        - provides an update method that act at a specifed site, indexed i. 
	            - returns a values estimating solution quality. when quality is stationnary within a criterion, the optimizer stops.
	        - provides a method "size" specifying at which site index to reverse sweep direction
			- provides a method "easy_index" that sepecify the best starting index.
	"""
	direction = 1
	cost = 1e6
	conv_val = 1e6
	iter_count = 0
	while conv_val >=crit:
		if iter_count > max_sweep:
			print(f"{updater.method_name()} failed to converge")
			break
		new_cost,direction = single_sweep(updater, direction)
		conv_val = convergence_eval(new_cost,cost)
		cost = new_cost
		iter_count += 1
	return updater.result()