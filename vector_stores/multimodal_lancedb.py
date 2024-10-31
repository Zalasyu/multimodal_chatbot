import lancedb


class MultiModalLanceDB:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def __enter__(self):
        """
        Connect to the LanceDB database and return self.

        This method is typically used with a `with` statement. For example:

        >>> with MultiModalLanceDB('path/to/db') as db:
        ...     # do something with the db
        ...     pass

        The database connection is automatically closed at the end of the `with` block.
        """
        self.db = lancedb.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Close the database connection.

        This method is called automatically at the end of the `with` block, so
        you don't need to call it explicitly. It is provided in case you want to
        close the database connection manually.
        """
        self.db.close()
