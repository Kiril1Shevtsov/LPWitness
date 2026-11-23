"""LPWitness: exact continuous-LP certificates, without trusting a solver."""
from .core import InputError, ResourceLimit, Problem, certificate, load_json, rational, verify

__all__ = ['InputError', 'ResourceLimit', 'Problem', 'certificate', 'load_json', 'rational', 'verify']
__version__ = '0.1.0'
