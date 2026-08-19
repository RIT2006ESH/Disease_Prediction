from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func

from app.db.session import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    disease_type = Column(String(20), nullable=False)
    input_data = Column(JSON, nullable=False)
    risk_label = Column(String(20), nullable=False)
    probability = Column(Float, nullable=False)
    top_features = Column(JSON, nullable=False)
    model_version = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())