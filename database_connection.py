import sqlalchemy as sa
import pandas as pd


def get_conection(user, password, host, port, db_name):
    """
    Creates an engine for a postgres data base
    :param user: str - user name
    :param password: str - user password
    :param host: str - host
    :param port: int - port
    :param db_name: str - data base name
    :return: engine
    """
    config = {
        "user": user,
        "password": password,
        "host": host,
        "port": port,
        "db_name": db_name
    }
    engine = sa.create_engine(
        'postgresql://{user}:{password}@{host}:{port}/{db_name}'.format(**config))
    return engine
      
    
def run_query_to_df(query_string, user, password, host, port, db_name):
    """
    Takes a query in SQL and returns a data frame of the results
    :param query_string: query in form of """"select * from table"""
    :return: pandas dataframe
    """
    engine = get_conection(user, password, host, port, db_name)
    query_string = query_string

    connection = engine.connect()

    df = pd.read_sql_query(query_string, connection)
    return df
