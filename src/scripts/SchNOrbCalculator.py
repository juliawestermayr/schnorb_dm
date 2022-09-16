from schnetpack.interfaces import SpkCalculator
from schnetpack import Properties
from schnetpack.md.calculators import MDCalculator
from schnorb.rotations import AimsRotator 
import ase.io
import numpy as np
import schnetpack as spk
import torch
from schnorb.data import SchNOrbProperties



class SchNOrbCalculator(SpkCalculator):
    """
    ASE calculator for schnorb machine learning models.

    Args:
        ml_model: Trained model for
            calculations
        device (str): select to run calculations on 'cuda' or 'cpu'
    """
    def __init__(self, model, required_properties=['energy', 'hamiltonian', 'overlap'],
                 force_handle='forces',
                 position_conversion=1.0,
                 force_conversion=1.0):
        super(SchNOrbCalculator, self).__init__(required_properties,force_handle,
                                        position_conversion,force_conversion)

        self.model = model

    def calculate(self, system):
        model_inputs = self.generate_input(system)
        # Call model
        model_results = self.model(model_inputs)


        #self._generate_input(system)
        self.results = {
            'energy': E,
            'hamiltonian': H,
            'overlap': S
        }
        self._update_system(system)

    def generate_input(self,system):
        #positions, atom_types, atom_masks = MDCalculator._get_system_molecules(MDCalculator,system)
        #neighbors, neighbor_mask = MDCalculator._get_system_neighbors(MDcalculator,system)
        at, properties = self.get_properties(idx)

        # get atom environment
        nbh_idx, offsets = self.environment_provider.get_environment(at)

        properties[SchNOrbProperties.neighbors] = torch.LongTensor(
            nbh_idx.astype(np.int))
        properties[SchNOrbProperties.cell_offset] = torch.FloatTensor(
            offsets.astype(np.float32))
        properties['_idx'] = torch.LongTensor(np.array([idx], dtype=np.int))


   
        return properties

class AtomsData(spk.data.AtomsData):

    def __init__(self, *args, add_rotations=False, rotator_cls=AimsRotator,
                 **kwargs):
        super(AtomsData, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.add_rotations = add_rotations
        self.rotator_cls = rotator_cls
    def __getitem__(self, inputpath):
        at = ase.io.read(inputpath)
        input={}
        input[SchNOrbProperties.R] = torch.FloatTensor([at.get_positions()])
        # get atom environment
        input[SchNOrbProperties.Z] = torch.LongTensor([at.get_atomic_numbers().astype(np.int)])
        input[SchNOrbProperties.cell_offset]=None
        input[SchNOrbProperties.cell]=None
        nbh_idx, offsets = self.environment_provider.get_environment(at)
        input[SchNOrbProperties.neighbors] = torch.LongTensor(
            [nbh_idx.astype(np.int)])
        input[SchNOrbProperties.cell_offset] = torch.FloatTensor(
            offsets.astype(np.float32))
        input['_idx'] = torch.LongTensor(np.array([0], dtype=np.int))
        # Calculate masks
        input["_atom_mask"] = torch.ones_like(input[SchNOrbProperties.Z]).float()
        mask = input[SchNOrbProperties.neighbors] >= 0
        input[SchNOrbProperties.neighbor_mask] = mask.float()
        input[SchNOrbProperties.neighbors] = (
            input[SchNOrbProperties.neighbors] * input[SchNOrbProperties.neighbor_mask].long()
        )

        return input
