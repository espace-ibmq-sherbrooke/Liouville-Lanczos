


from typing import Tuple

from qiskit.pulse.exceptions import PulseError
from qiskit.transpiler.passes import (
    EchoRZXWeylDecomposition,
)

class ECREchoRZXWeylDecomposition(EchoRZXWeylDecomposition):
    def _is_native(self, qubit_pair: Tuple) -> bool:
        """Return the direction of the qubit pair that is native, i.e. with the shortest schedule.
           For ECR based machine return the available direction."""
        try:
            return super()._is_native(qubit_pair) 
        except:
            try:
                ec1 = self._inst_map.get("ecr", qubit_pair)
            except:
                ec1 = None
            try:
                ec2 = self._inst_map.get("ecr", qubit_pair[::-1])
            except:
                ec2 = None
            if not (ec1 or ec2):
                raise PulseError(f"Operation 'ecr' is not defined for qubit pair {qubit_pair}.")
            if (ec1 is not None) and (ec2 is not None):
                return ec1.duration < ec2.duration
            return (ec1 is not None)