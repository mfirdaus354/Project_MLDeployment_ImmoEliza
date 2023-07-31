from sqlalchemy import Column, Float, String, Integer, Boolean, ForeignKey
from sqlalchemy.orm import relationship

from .database import Base


class Simulation(Base):
    __tablename__ = "simulations"
    simulation_id = Column(Integer, primary_key=True, unique=True, index=True)
    predicted_price = Column(Integer)
    input_id = Column(Integer, ForeignKey("items.id"))
    input_data = relationship("Item", back_populates="simulation")


class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    plot_area = Column(Float)
    habitable_surface = Column(Float)
    land_surface = Column(Float)
    bedroom_count = Column(Integer)
    room_count = Column(Integer)
    simulation = relationship("Simulation", back_populates="input_data")
