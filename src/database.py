import os
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_intent: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    suggested_reply: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def save_prediction(
    text: str,
    predicted_intent: str,
    confidence: float,
    suggested_reply: str,
) -> Prediction:
    with SessionLocal() as session:
        prediction = Prediction(
            text=text,
            predicted_intent=predicted_intent,
            confidence=confidence,
            suggested_reply=suggested_reply,
        )
        session.add(prediction)
        session.commit()
        session.refresh(prediction)
        return prediction


def get_recent_predictions(limit: int) -> list[Prediction]:
    with SessionLocal() as session:
        statement = select(Prediction).order_by(Prediction.created_at.desc()).limit(limit)
        return list(session.scalars(statement))
