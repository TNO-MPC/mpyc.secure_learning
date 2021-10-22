"""
Test utils for MPyC
"""
from typing import Type, Union, cast, overload

import numpy as np
from mpyc.runtime import mpc
from mpyc.sectypes import SecureFixedPoint

from tno.mpc.mpyc.secure_learning.utils import (
    Matrix,
    NumpyNumberArray,
    NumpyObjectArray,
    SecureObjectType,
    Vector,
)


@overload
def mpyc_input(data: float, secfxp: Type[SecureFixedPoint]) -> SecureFixedPoint:
    ...


@overload
def mpyc_input(
    data: Vector[float], secfxp: Type[SecureFixedPoint]
) -> Vector[SecureFixedPoint]:
    ...


@overload
def mpyc_input(
    data: Matrix[float], secfxp: Type[SecureFixedPoint]
) -> Matrix[SecureFixedPoint]:
    ...


@overload
def mpyc_input(
    data: Union[NumpyNumberArray], secfxp: Type[SecureFixedPoint]
) -> NumpyObjectArray:
    ...


def mpyc_input(
    data: Union[float, Vector[float], Matrix[float], NumpyNumberArray],
    secfxp: Type[SecureFixedPoint],
) -> Union[
    SecureFixedPoint,
    Vector[SecureFixedPoint],
    Matrix[SecureFixedPoint],
    NumpyObjectArray,
]:
    """
    Converts data to secfxp type and shares it over all parties in the
    MPyC protocol.

    :param data: Input data
    :param secfxp: A SecureFixedPoint type
    :return: Inputted data of the same type
    """
    secfxp_data = _convert_to_secfxp(data, secfxp=secfxp)
    return _mpyc_input(secfxp_data)


@overload
def _convert_to_secfxp(data: float, secfxp: Type[SecureFixedPoint]) -> SecureFixedPoint:
    ...


@overload
def _convert_to_secfxp(
    data: Vector[float], secfxp: Type[SecureFixedPoint]
) -> Vector[SecureFixedPoint]:
    ...


@overload
def _convert_to_secfxp(
    data: Matrix[float], secfxp: Type[SecureFixedPoint]
) -> Matrix[SecureFixedPoint]:
    ...


@overload
def _convert_to_secfxp(
    data: NumpyNumberArray, secfxp: Type[SecureFixedPoint]
) -> NumpyObjectArray:
    ...


def _convert_to_secfxp(
    data: Union[float, Vector[float], Matrix[float], NumpyNumberArray],
    secfxp: Type[SecureFixedPoint],
) -> Union[
    SecureFixedPoint,
    Vector[SecureFixedPoint],
    Matrix[SecureFixedPoint],
    NumpyObjectArray,
]:
    """
    Convert numbers (in a list or numpy array) to secure numbers (in the
    same format).

    :param data: Input data
    :param secfxp: A SecureFixedPoint type
    :return: Converted data
    """
    if isinstance(data, list):
        if isinstance(data[0], list):
            return [
                list(map(lambda _: _convert_to_secfxp(_, secfxp=secfxp), row))
                for row in cast(Matrix[float], data)
            ]
        return list(
            map(
                lambda _: _convert_to_secfxp(_, secfxp=secfxp),
                cast(Vector[float], data),
            )
        )
    if isinstance(data, np.ndarray):
        return cast(
            NumpyObjectArray,
            np.vectorize(lambda _: secfxp(float(_), integral=False))(data),
        )
    return secfxp(data, integral=False)


@overload
def _mpyc_input(data: SecureObjectType) -> SecureObjectType:
    ...


@overload
def _mpyc_input(data: Vector[SecureObjectType]) -> Vector[SecureObjectType]:
    ...


@overload
def _mpyc_input(data: Matrix[SecureObjectType]) -> Matrix[SecureObjectType]:
    ...


@overload
def _mpyc_input(data: NumpyObjectArray) -> NumpyObjectArray:
    ...


def _mpyc_input(  # type: ignore[misc]
    data: Union[
        SecureObjectType,
        Vector[SecureObjectType],
        Matrix[SecureObjectType],
        NumpyObjectArray,
    ]
) -> Union[
    SecureObjectType,
    Vector[SecureObjectType],
    Matrix[SecureObjectType],
    NumpyObjectArray,
]:
    """
    MPyC mpc.input function for several data types.

    :param data: Input data with SecureObject elements.
    :return: Inputted data of the same type
    """
    if isinstance(data, np.ndarray):
        if len(data.shape) > 1:
            return np.array([_mpyc_input(_) for _ in data])
        return np.array(mpc.input(data.tolist(), senders=0))
    if isinstance(data, list) and isinstance(data[0], list):
        return cast(Matrix[SecureObjectType], [_mpyc_input(vector) for vector in data])
    return cast(
        Union[SecureObjectType, Vector[SecureObjectType]], mpc.input(data, senders=0)
    )


@overload
async def mpyc_output(data: SecureObjectType) -> float:
    ...


@overload
async def mpyc_output(data: Vector[SecureObjectType]) -> Vector[float]:
    ...


@overload
async def mpyc_output(data: Matrix[SecureObjectType]) -> Matrix[float]:
    ...


@overload
async def mpyc_output(data: NumpyObjectArray) -> NumpyObjectArray:
    ...


async def mpyc_output(  # type: ignore[misc]
    data: Union[
        SecureObjectType,
        Vector[SecureObjectType],
        Matrix[SecureObjectType],
        NumpyObjectArray,
    ]
) -> Union[float, Vector[float], Matrix[float], NumpyObjectArray]:
    """
    MPyC mpc.output function for several data types.

    :param data: Output data in with SecureObject elements.
    :return: Output data of the same type
    """
    if isinstance(data, np.ndarray):
        if len(data.shape) > 1:
            return np.array([await mpyc_output(_) for _ in data])
        return np.array(await mpc.output(data.tolist()))
    if isinstance(data, list) and isinstance(data[0], list):
        return [await mpyc_output(_) for _ in data]
    return cast(
        Union[float, Vector[float]],
        await mpc.output(cast(Union[SecureObjectType, Vector[SecureObjectType]], data)),
    )
