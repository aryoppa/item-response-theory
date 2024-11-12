import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

connection_string = "mysql+mysqlconnector://root:@localhost:3306/test"
csv_file_path = 'data/raw/cobi.csv'

try:
    # Create SQLAlchemy engine
    engine = create_engine(connection_string)
    
    # Read CSV into DataFrame
    df = pd.read_csv(csv_file_path)
    print("Dataframe loaded successfully:")
    print(df)
    
    # Write DataFrame to MySQL database
    df.to_sql(con=engine, name="quiz_questions", if_exists="append", index=False)
    print("Data written to database successfully.")
except SQLAlchemyError as e:
    print("Error while writing to the database:")
    print(str(e))
except Exception as e:
    print("General error:")
    print(str(e))
