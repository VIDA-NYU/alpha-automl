from sqlalchemy import Column
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
import uuid


# From http://docs.sqlalchemy.org/en/latest/core/custom_types.html
# #backend-agnostic-guid-type

class UUID(TypeDecorator):
    """Platform-independent UUID type.

    Uses PostgreSQL's UUID type, otherwise uses
    CHAR(32), storing as stringified hex values.

    """
    impl = CHAR

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_UUID())
        else:
            return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return "%.32x" % uuid.UUID(value).int
            else:
                # hexstring
                return "%.32x" % value.int

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                value = uuid.UUID(value)
            return value


class UuidMixin(object):
    """A mixin that adds a UUID column and sets it at construction time.

    Using the ``default=`` keyword on the Column would only set the attribute
    at insert time.
    """
    id = Column(UUID, primary_key=True, default=uuid.uuid4)

    def __init__(self, **kwargs):
        super(UuidMixin, self).__init__(**kwargs)
        if 'id' not in kwargs:
            self.id = uuid.uuid4()
