import pypyodbc as odbc
import pandas as pd
import requests
import json


async def get_list_examTypeName():
    DRIVER_NAME='SQL SERVER'
    SERVER_NAME='.'
    DATABASE_NAME='HelthcareAssistant'

    connection_string=f"""
      DRIVER={{{DRIVER_NAME}}};
      SERVER={SERVER_NAME};
      DATABASE={DATABASE_NAME};
      Trust_Connection=yes;
    """
    connexion=odbc.connect(connection_string)
    query = ("SELECT examTypeName FROM Exam_Type")
    data = pd.read_sql(query, connexion)
    connexion.close()
    list_of_examTypeName_arrays = data.to_json(orient="values")
    json_data = json.loads(list_of_examTypeName_arrays)
    list_of_strings = [item[0] for item in json_data]
    result_string = ', '.join(list_of_strings)
    return result_string  
async def get_Vocabs(ExamTypeName:str):
    DRIVER_NAME='SQL SERVER'
    SERVER_NAME='.'
    DATABASE_NAME='HelthcareAssistant'

    connection_string=f"""
      DRIVER={{{DRIVER_NAME}}};
      SERVER={SERVER_NAME};
      DATABASE={DATABASE_NAME};
      Trust_Connection=yes;
    """
    connexion=odbc.connect(connection_string)
    query = ("SELECT top 1 CustomeVocabulary FROM Exam_Type WHERE examTypeName='"+ExamTypeName+"'" )
    data = pd.read_sql(query, connexion)
    connexion.close()
    Vocabs = data.to_json(orient="values")
    json_data = json.loads(Vocabs)
    list_Vocabs = [item[0] for item in json_data]
    ListOfVocabs = ', '.join(list_Vocabs)
    return ListOfVocabs
