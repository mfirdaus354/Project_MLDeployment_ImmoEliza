from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from immo_eliza_sql_app import models 
from immo_eliza_sql_app.schemas import Item, ItemBase, ItemCreate, Simulation, SimulationBase, SimulationCreate
from immo_eliza_sql_app.database import engine, SessionLocal
from immo_eliza_sql_app.crud import create_item, create_simulation
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
@app.post("/insert-data", response_model=Item)
async def insert_data(
    item: ItemCreate, 
    db: Session = Depends(get_db)
    ):
    return create_item(db=db, item=item)

# @app.get("/database/get-datas/{start}/{stop}", response_model=models.Item)
# async def get_data(
#     start: int,
#     stop: int,
#     db: Session = Depends(get_db)
#     ):
#     return crud.get_items(db:db, skip=start, limit=stop)

