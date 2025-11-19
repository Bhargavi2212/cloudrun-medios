"""
Base repository class with common CRUD operations.
"""

from typing import Generic, TypeVar, Type, List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func
from sqlalchemy.orm import selectinload

from ..base import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Base repository with common async CRUD operations."""
    
    def __init__(self, model: Type[ModelType], session: AsyncSession):
        """
        Initialize repository.
        
        Args:
            model: SQLAlchemy model class
            session: Async database session
        """
        self.model = model
        self.session = session
    
    async def create(self, **kwargs) -> ModelType:
        """
        Create a new record.
        
        Args:
            **kwargs: Model field values
            
        Returns:
            Created model instance
        """
        instance = self.model(**kwargs)
        self.session.add(instance)
        await self.session.commit()
        await self.session.refresh(instance)
        return instance
    
    async def get_by_id(self, id_value: Any) -> Optional[ModelType]:
        """
        Get record by primary key.
        
        Args:
            id_value: Primary key value
            
        Returns:
            Model instance or None
        """
        return await self.session.get(self.model, id_value)
    
    async def get_by_field(self, field_name: str, field_value: Any) -> Optional[ModelType]:
        """
        Get record by specific field.
        
        Args:
            field_name: Field name to search by
            field_value: Field value to match
            
        Returns:
            Model instance or None
        """
        field = getattr(self.model, field_name)
        result = await self.session.execute(
            select(self.model).where(field == field_value)
        )
        return result.scalar_one_or_none()
    
    async def get_all(
        self, 
        skip: int = 0, 
        limit: int = 100,
        order_by: Optional[str] = None
    ) -> List[ModelType]:
        """
        Get all records with pagination.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            order_by: Field name to order by
            
        Returns:
            List of model instances
        """
        query = select(self.model).offset(skip).limit(limit)
        
        if order_by:
            order_field = getattr(self.model, order_by, None)
            if order_field is not None:
                query = query.order_by(order_field.desc())
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def update(self, id_value: Any, **kwargs) -> Optional[ModelType]:
        """
        Update record by primary key.
        
        Args:
            id_value: Primary key value
            **kwargs: Fields to update
            
        Returns:
            Updated model instance or None
        """
        # Get primary key field name
        pk_field = list(self.model.__table__.primary_key.columns)[0]
        
        await self.session.execute(
            update(self.model)
            .where(pk_field == id_value)
            .values(**kwargs)
        )
        await self.session.commit()
        
        return await self.get_by_id(id_value)
    
    async def delete(self, id_value: Any) -> bool:
        """
        Delete record by primary key.
        
        Args:
            id_value: Primary key value
            
        Returns:
            True if deleted, False if not found
        """
        # Get primary key field name
        pk_field = list(self.model.__table__.primary_key.columns)[0]
        
        result = await self.session.execute(
            delete(self.model).where(pk_field == id_value)
        )
        await self.session.commit()
        
        return result.rowcount > 0
    
    async def count(self, **filters) -> int:
        """
        Count records with optional filters.
        
        Args:
            **filters: Field filters
            
        Returns:
            Number of matching records
        """
        query = select(func.count(self.model.id))
        
        for field_name, field_value in filters.items():
            field = getattr(self.model, field_name, None)
            if field is not None:
                query = query.where(field == field_value)
        
        result = await self.session.execute(query)
        return result.scalar() or 0
    
    async def exists(self, **filters) -> bool:
        """
        Check if record exists with given filters.
        
        Args:
            **filters: Field filters
            
        Returns:
            True if record exists
        """
        count = await self.count(**filters)
        return count > 0