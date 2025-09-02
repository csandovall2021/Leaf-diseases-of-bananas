import mysql.connector
from mysql.connector import Error
import os
from contextlib import contextmanager

# Configuración de la base de datos
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),  # Cambia por tu contraseña
    'database': os.getenv('DB_NAME', 'FitoApp'),
    'charset': 'utf8mb4',
    'collation': 'utf8mb4_unicode_ci',
    'autocommit': True
}

class DatabaseConnection:
    def __init__(self):
        self.connection = None
    
    def connect(self):
        """Establecer conexión con la base de datos"""
        try:
            if self.connection is None or not self.connection.is_connected():
                self.connection = mysql.connector.connect(**DB_CONFIG)
                print("Conexión exitosa a la base de datos FitoApp.")
            return self.connection
        except Error as e:
            print(f"Error al conectar con la base de datos: {e}")
            return None
    
    def disconnect(self):
        """Cerrar la conexión con la base de datos"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Conexión cerrada.")
    
    def is_connected(self):
        """Verificar si la conexión está activa"""
        return self.connection and self.connection.is_connected()
    
    def execute_query(self, query, params=None, fetch=False):
        """Ejecutar una consulta en la base de datos"""
        try:
            connection = self.connect()
            if connection:
                cursor = connection.cursor()
                cursor.execute(query, params or ())
                
                if fetch:
                    result = cursor.fetchall()
                    cursor.close()
                    return result
                else:
                    connection.commit()
                    cursor.close()
                    return True
        except Error as e:
            print(f"Error al ejecutar consulta: {e}")
            return None
    
    def get_cursor(self):
        """Obtener un cursor para operaciones complejas"""
        connection = self.connect()
        if connection:
            return connection.cursor()
        return None

# Instancia global de la conexión
db = DatabaseConnection()

@contextmanager
def get_db_connection():
    """Context manager para manejo seguro de conexiones"""
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        yield connection
    except Error as e:
        print(f"Error de base de datos: {e}")
        if connection:
            connection.rollback()
        raise
    finally:
        if connection and connection.is_connected():
            connection.close()

def test_connection():
    """Función para probar la conexión"""
    try:
        connection = db.connect()
        if connection:
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            print("Test de conexión exitoso.")
            return True
    except Error as e:
        print(f"Error en test de conexión: {e}")
        return False

def create_tables():
    """Crear las tablas necesarias si no existen"""
    queries = [
        """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            image_path VARCHAR(255),
            prediction_result VARCHAR(100),
            confidence DECIMAL(5,4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS diseases (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            description TEXT,
            treatment TEXT,
            prevention TEXT
        )
        """
    ]
    
    for query in queries:
        result = db.execute_query(query)
        if result:
            print("Tabla creada o ya existe.")

# Ejemplo de uso
if __name__ == "__main__":
    # Probar la conexión
    test_connection()
    
    # Crear tablas
    create_tables()
    
    # Cerrar conexión
    db.disconnect()