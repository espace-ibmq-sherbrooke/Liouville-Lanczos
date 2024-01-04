"""
    Liouville-Lanczos: A library for Many-Body Green's function on quantum and classical computer.
    Copyright (C) 2024  Alexandre Foley

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import quimb.tensor as qtn
import numpy as np


def generate_id_MPO(upper_ind: str, lower_ind: str, Nsite: int, factor: float = 1):
	id = [np.array([[np.eye(2) * factor]]) for i in range(Nsite)]
	id[0] = id[0][0, :, :, :]
	id[-1] = id[0]
	return qtn.MatrixProductOperator(id, upper_ind_id=upper_ind, lower_ind_id=lower_ind)


def generate_swap_MPO(
	upper_ind: str, lower_ind: str, Nsite: int, swap_sites: list[int] = [0, -1]
):
	if swap_sites[1] < 0:
		swap_sites[1] += Nsite
	if swap_sites[0] < 0:
		swap_sites[0] += Nsite
	id = np.array([[np.eye(2)]])
	transmit = np.eye(8, 8).reshape(4, 2, 4, 2).transpose(0, 2, 1, 3)
	X, iY, Z, I = (
		np.array([[0, 1], [1, 0]]),
		np.array([[0, 1], [-1, 0]]),
		np.array([[1, 0], [0, -1]]),
		np.array([[1, 0], [0, 1]]),
	)
	Lswap = np.zeros((1, 4, 2, 2))
	Lswap[0, 0, :, :] = iY / 2
	Lswap[0, 1, :, :] = X / 2
	Lswap[0, 2, :, :] = Z / 2
	Lswap[0, 3, :, :] = I / 2
	Rswap = np.zeros((4, 1, 2, 2))
	Rswap[0, 0, :, :] = -iY
	Rswap[1, 0, :, :] = X
	Rswap[2, 0, :, :] = Z
	Rswap[3, 0, :, :] = I

	def select(i):
		if i < swap_sites[0]:
			return id
		if i == swap_sites[0]:
			return Lswap
		if i < swap_sites[1]:
			return transmit
		if i == swap_sites[1]:
			return Rswap
		return id

	op = [select(i) for i in range(Nsite)]
	op[0] = op[0][0, :, :, :]
	op[-1] = op[-1][:, 0, :, :]
	return qtn.MatrixProductOperator(op, upper_ind_id=upper_ind, lower_ind_id=lower_ind)
