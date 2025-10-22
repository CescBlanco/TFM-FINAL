from pydantic import BaseModel
from typing import List

class UserData(BaseModel):
    Edad: int
    Sexo_Mujer: bool
    UsoServiciosExtra: bool
    ratio_cantidad_2025_2024: float
    Diversidad_servicios_extra: int
    TienePagos: bool
    TotalVisitas: int
    DiasActivo: int
    VisitasUlt90: int
    VisitasUlt180: int
    TieneAccesos: bool
    VisitasPrimerTrimestre: int
    VisitasUltimoTrimestre: int
    DiaFav_domingo: bool
    DiaFav_jueves: bool
    DiaFav_lunes: bool
    DiaFav_martes: bool
    DiaFav_miercoles: bool
    DiaFav_sabado: bool
    DiaFav_viernes: bool
    EstFav_invierno: bool
    EstFav_otono: bool
    EstFav_primavera: bool
    EstFav_verano: bool


class MultiUserData(BaseModel):
    datos: List[UserData]

class IDRequest(BaseModel):
    IdPersona: int

class IDListRequest(BaseModel):
    Ids: List[int]