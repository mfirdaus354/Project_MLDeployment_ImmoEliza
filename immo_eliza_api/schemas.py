from pydantic import BaseModel


class ItemBase(BaseModel):
    plot_area: float
    habitable_surface: float
    land_surface: float
    bedroom_count: int
    room_count: int


class ItemCreate(ItemBase):
    pass


class Item(ItemBase):
    id: int

    class Config:
        orm_mode = True


class SimulationBase(BaseModel):
    predicted_price: int


class SimulationCreate(SimulationBase):
    pass


class Simulation(SimulationBase):
    id: int
    input_id: int

    class Config:
        orm_mode = True
