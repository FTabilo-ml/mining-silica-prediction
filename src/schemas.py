# src/schemas.py
from typing import Dict, List
from pydantic import BaseModel, RootModel

# Payload de una sola fila: diccionario {feature: valor}
class Features(RootModel[Dict[str, float]]):
    @property
    def as_dict(self) -> Dict[str, float]:
        return self.root

# Payload por lotes: lista de diccionarios
class BatchFeatures(BaseModel):
    rows: List[Dict[str, float]]
