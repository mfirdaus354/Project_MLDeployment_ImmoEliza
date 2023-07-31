from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from . import crud, models, schemas
from .database import SessionLocal, engine
from .schemas import ItemBase, ItemCreate, Item, SimulationBase, SimulationCreate, Simulation
from .models import Simulation, Item
# from .
models.Base.metadata.create_all(bind=engine)

app = FastAPI()


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# GET REQUESTS
@app.get("/")
async def root():
    intro = {
        "message": "This is the API for Project-MLOps-Deployment"
    }
    return intro


# API endpoint to enter new data
@app.post("/database/insert-data/plot_area={plot_area}", response_model=schemas.ItemBase)
async def insert_data(
    plot_area: schemas.ItemCreate = 0, 
    db: Session = Depends(get_db)
    ):
    db_simulation_instance = crud.get_simulation(
        db=db,  
        simulation_id=Item.simulation_id)
    if db_simulation_instance:
        raise HTTPException(status_code=400, detail="This simulation has been executed. You cannot create another simulation on top of this simulation id")
    
    item = schemas.ItemCreate(plot_area=plot_area)
    return crud.create_item(db=db, item=item)

# @app.get("/database/get-datas/{start}/{stop}", response_model=models.Item)
# async def get_data(
#     start: int,
#     stop: int,
#     db: Session = Depends(get_db)
#     ):
#     return crud.get_items(db:db, skip=start, limit=stop)

