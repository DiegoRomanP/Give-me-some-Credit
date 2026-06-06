from pydantic import BaseModel, Field
from typing import Optional

class CreditPredictionRequest(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ..., description="Saldo total en tarjetas/líneas de crédito dividido por límites", ge=0.0
    )
    age: int = Field(..., description="Edad del prestatario", ge=0)
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ..., alias="NumberOfTime30-59DaysPastDueNotWorse", ge=0
    )
    DebtRatio: float = Field(..., description="Ratio de endeudamiento", ge=0.0)
    
    # Optional porque nuestro modelo MICE sabe cómo imputarlos
    MonthlyIncome: Optional[float] = Field(None, description="Ingreso mensual bruto", ge=0.0)
    
    NumberOfOpenCreditLinesAndLoans: int = Field(..., description="Préstamos abiertos y líneas de crédito", ge=0)
    NumberOfTimes90DaysLate: int = Field(..., description="Veces con mora de 90 días o más", ge=0)
    NumberRealEstateLoansOrLines: int = Field(..., description="Préstamos hipotecarios", ge=0)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ..., alias="NumberOfTime60-89DaysPastDueNotWorse", ge=0
    )
    
    # Optional para permitir imputación
    NumberOfDependents: Optional[int] = Field(None, description="Número de dependientes", ge=0)

class CreditPredictionResponse(BaseModel):
    default_probability: float = Field(..., description="Probabilidad de incumplimiento en los próximos 2 años")
    risk_level: str = Field(..., description="Clasificación del riesgo (Bajo, Medio, Alto)")