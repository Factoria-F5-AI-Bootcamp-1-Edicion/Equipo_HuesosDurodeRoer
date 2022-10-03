#Librerias para trabajar con la carga del modelo y el objeto transformer
import joblib
import lightgbm

import pandas as pd


def paciente():
    '''Pide al usuario-doctor que ingrese los datos del nuevo paciente por teclado'''  
    #Mensaje de bienvenida
    print("¡Hola! Introduce los datos del nuevo paciente")

    #Escribimos genero
    gender = input("Por favor ingrese el genero del paciente (Male/Female): ")

    #Escribimos work_type
    work_type = input("\nPor favor ingrese el tipo de trabajo(Private/Self-employed/Govt_job/children): \n")

    ##Leemos Residence_type
    residence_type = input("\nPor favor ingrese el tipo de residencia(Urban/Rural): \n")

    ##Leemos smoking_status
    smoking_status = input("\nPor favor ingrese el tipo de fumador(formerly smoked/never smoked/smokes/Unknown): \n")

    ##Leemos age
    age = input("\nPor favor ingrese la edad del pàciente: \n")

    ##Leemos hypertension
    hypertension = input("\nPor favor ingrese la hipertension(1 or 0): \n")

    ##Leemos heart_disease
    heart_disease = input("\nPor favor ingrese si esta enfermo del corazón(1 or 0): \n")

    ##Leemos ever_married
    ever_married = input("\nPor favor ingrese si esta casado (Yes/No): \n")

    ##Leemos avg_glucose_level
    avg_glucose_level = input("\nPor favor ingrese nivel medio de glucosa: \n")

    ##Leemos avg_glucose_level
    bmi = input("\nPor favor ingrese el BMI (Base Muscle Index): \n")
    '''
    gender = 'Male'
    age = '80'
    hypertension = '0'
    heart_disease = '0'
    ever_married = 'Yes'
    work_type = 'Private'
    residence_type = 'Rural'
    avg_glucose_level = '174.12'
    bmi = '25'
    smoking_status = 'smokes'
    '''
    #Age será un entero o binario (0 ó 1)
    age = int(age)
    #BMI, avg_glucose_level será un real, así que usamos float()
    bmi = float(bmi)
    avg_glucose_level = float(avg_glucose_level)
    #Bool
    heart_disease = int(heart_disease)
    hypertension = int(hypertension)

    lista_variables_predictoras = [[gender, age, hypertension, heart_disease,  ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]]

    return(lista_variables_predictoras)

    #Llamo a mi funcion predictora
    #predict(variables_predictoras)

list_variables_predictoras = paciente()

print(list_variables_predictoras)

columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
X_valid = pd.DataFrame(list_variables_predictoras, columns = columns)

print(X_valid)

#Carga el transformer
transformer = joblib.load('transformer.pkl')

#Me falta las transformaciones de X_test
X_valid = transformer.transform(X_valid)

#Carga el modelo lightgbm y predice
mod = lightgbm.Booster(model_file='lgbr_hyper_os.txt')
y_predict = mod.predict(X_valid)
print("Stroke: ", (y_predict * 100), "%")
