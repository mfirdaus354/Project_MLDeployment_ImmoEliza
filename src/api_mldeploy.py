from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel

# Replace 'your_database_url' with your actual database URL
DATABASE_URL = "sqlite:///./test.db"

# Create the engine and connect to the database
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a base class for declarative models
Base = declarative_base()

# Define the APIDatabase model
class APIDatabase(Base):
    __tablename__ = "APIDatabase"
    id = Column(Integer, primary_key=True, index=True)
    simulation_id = Column(String, index=True, unique=True)
    price = Column(Integer, nullable=False)
    room_count = Column(Integer, nullable=False)
    bedroom_count = Column(Integer, nullable=False)
    habitable_surface = Column(Float, nullable=False)
    land_surface = Column(Float, nullable=False)
    plot_area = Column(Float, nullable=False)

# Create the database tables
Base.metadata.create_all(bind=engine)

# Create the FastAPI app
app = FastAPI()

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Input template for user data
class InputTemplate(BaseModel):
    simulation_id: str
    price: int
    room_count: int
    bedroom_count: int
    habitable_surface: float
    land_surface: float
    plot_area: float

# API endpoint to create a new entry in the APIDatabase
@app.post("/api/", response_model=APIDatabase)
def create_item(item: InputTemplate, db: Session = Depends(get_db)):
    db_item = APIDatabase(**item.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

# API endpoint to get an item from the APIDatabase by its ID
@app.get("/api/{simulation_id}", response_model=APIDatabase)
def read_item(simulation_id: str, db: Session = Depends(get_db)):
    item = db.query(APIDatabase).filter(APIDatabase.simulation_id == simulation_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

# API endpoint to get all items from the APIDatabase
@app.get("/api/", response_model=list[APIDatabase])
def read_all_items(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    items = db.query(APIDatabase).offset(skip).limit(limit).all()
    return items

# API endpoint to update an item in the APIDatabase by its ID
@app.put("/api/{simulation_id}", response_model=APIDatabase)
def update_item(simulation_id: str, item: InputTemplate, db: Session = Depends(get_db)):
    db_item = db.query(APIDatabase).filter(APIDatabase.simulation_id == simulation_id).first()
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    for key, value in item.dict().items():
        setattr(db_item, key, value)
    db.commit()
    db.refresh(db_item)
    return db_item

# API endpoint to delete an item from the APIDatabase by its ID
@app.delete("/api/{simulation_id}")
def delete_item(simulation_id: str, db: Session = Depends(get_db)):
    db_item = db.query(APIDatabase).filter(APIDatabase.simulation_id == simulation_id).first()
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    db.delete(db_item)
    db.commit()
    return {"message": "Item deleted successfully"}
