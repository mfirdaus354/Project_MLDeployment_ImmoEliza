from sqlalchemy.orm import Session

from . import models, schemas

from icecream import install

install()


def get_simulation(db: Session, simulation_id: int):
    return (
        db.query(models.Simulation)
        .filter(models.Simulation.id == simulation_id)
        .first()
    )


def get_simulations(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Simulations).offset(skip).limit(limit).all()


def create_simulation(db: Session, simulation: schemas.SimulationCreate, item_id: int):
    db_simulation = models.Simulation(
        predicted_price=simulation.predicted_price, input_id=item_id
    )
    db.add(db_simulation)
    db.commit()
    db.refresh(db_simulation)
    return db_simulation


def get_item(db: Session, item_id: int):
    return db.query(models.Item).filter(models.Item.id == item_id).first()


def get_items(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Item).offset(skip).limit(limit).all()


def create_item(db: Session, item: schemas.ItemCreate):
    db_item = models.Item(**item.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item


def get_items_without_sim(db: Session, skip: int = 0, limit: int = 100):
    subq = db.query(models.Simulation).with_entities(models.Simulation.input_id)
    return db.query(models.Item).filter(~models.Item.id.in_(subq)).all()
