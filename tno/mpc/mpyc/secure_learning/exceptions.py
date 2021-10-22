"""
Defines custom exceptions for the tno.mpc.mpyc.secure_learning library.
"""


class BaseSecureLearnError(Exception):
    """
    Base class for custom exceptions in the tno.mpc.mpyc.secure_learning library
    """


class MissingFunctionError(BaseSecureLearnError):
    """
    The function has not been initialized.
    """


class SecureLearnTypeError(BaseSecureLearnError, TypeError):
    """
    Received object of unexpected type.
    """


class SecureLearnUninitializedSolverError(BaseSecureLearnError):
    """
    Solver has not been fully initialized.
    """


class SecureLearnValueError(BaseSecureLearnError, ValueError):
    """
    Received object with unexpected value.
    """


class UnknownPenaltyError(BaseSecureLearnError):
    """
    Penalty is not recognized.
    """
