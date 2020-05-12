# -*- coding: utf-8 -*-
"""
@author: Amador Lopez & Federico Valdez & Antonio Alcantara
"""



# --- Load libraries -----------------------------------------------------------
from flask import Flask
from flask import Flask, flash, redirect, render_template, request, session, abort
import os
import requests
import pickle
import pandas as pd
from pandas import DataFrame
from flask_mobility import Mobility
from flask_mobility.decorators import mobile_template
from flask_mobility.decorators import mobilized
import random
from random import randint

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = preprocessing.MinMaxScaler()



# --- Load pickle objects ------------------------------------------------------
# min_max_scaler = pickle.load(open('./pkls/min_max_scaler.pkl', 'rb'))
# df1 = pickle.load(open('./pkls/df1.pkl', 'rb'))
# df1_normalized = pickle.load(open('./pkls/df1_normalized.pkl', 'rb'))
# best_model_cargado = pickle.load(open('./pkls/best_model_calidad.pkl', 'rb'))


df1 = pickle.load(open('./pkls/df1_multi.pkl', 'rb'))
min_max_scaler_cargado = pickle.load(open('./pkls/min_max_scaler_multi.pkl', 'rb'))
# min_max_scaler = pickle.load(open('./pkls/scaled_array_multi.pkl', 'rb'))
# normalizar = pickle.load(open('./pkls/normalizar_multi.pkl', 'rb'))
df1_normalized = pickle.load(open('./pkls/df1_normalized_multi.pkl', 'rb'))
best_model_cargado = pickle.load(open('./pkls/clf.pkl', 'rb'))
# desnormalizar = pickle.load(open('./pkls/desnormalizar_multi.pkl', 'rb'))
# desnormalizar_pred = pickle.load(open('./pkls/desnormalizar_pred_multi.pkl', 'rb'))
defaults = pickle.load(open('./pkls/input_ranges.pkl', 'rb'))
stats = pickle.load(open('./pkls/minmax_stats.pkl', 'rb'))


# --- Definir funciones -------------------------------------------------------

# def refresh_values():
#     random.seed()
#     value_dict = [randint(1, 8000), randint(1, 8000), randint(1, 2000), randint(1, 2000)]
#     return value_dict
#
#     # ############################ PI REALTIME
#     # # Set auth, Define username, password of the account to be used
#     # username = "pidemo"
#     # password = ""
#     #
#     # # PI Data Archive
#     # piServers = PIServers()
#     # piServer = piServers['pitepeaca']
#     # credential = NetworkCredential(username, password)
#     # piServer.Connect(credential)
#     #
#     # # Tags to extract: HeatPetCoke	HeatRefuseFuel	VTIid06	VTIid07
#     # tags = ['MX-TEP-LHV-112002-DNA-88601-01-CMP-M00-QM',
#     #         'MX-TEP-LHV-135002-DNA-88601-01-CMP-M00-QM',
#     #         'SC_53331.TPC','SC_53631.TPC']
#     #
#     # value_dict = []
#     # for tag in tags:
#     #     pt = PIPoint.FindPIPoint(piServer, tag)
#     #     name = pt.Name.lower()
#     #     # print('Collecting:' + tag)
#     #     current_value = pt.CurrentValue()
#     #     rounded = round(current_value.Value,1)
#     #     value_dict.append(rounded)
#
#     # print(value_dict)
#     return value_dict


def normalizar(mi_array):
    mi_dataframe_norm = min_max_scaler_cargado.transform(mi_array)
    # list_norm = mi_dataframe_norm.tolist()
    return mi_dataframe_norm


def desnormalizar(row9_df):
    float_array2_normalized = row9_df.values.astype(float)
    inversed = min_max_scaler_cargado.inverse_transform(float_array2_normalized)

    cols = df1_normalized.columns.tolist()
    inversed = pd.DataFrame(inversed, columns= cols)
    return(inversed)


def desnormalizar_pred(mi_array_pred):
    mi_array_pred_df = pd.DataFrame(mi_array_pred, columns= ['Pred_KcalxKg',
       'Pred_Quality_CalLibre', 'Pred_TempAireSec', 'Pred_PresCaperuza', 'Pred_PresComp3',
       'Pred_PresComp4', 'Pred_SO3OTE', 'Pred_SO3PTE', 'Pred_SO3Clk', 'Pred_C3AClk', 'Pred_C3SClk', 'Pred_FSCClk',
       'Pred_CloroOTE', 'Pred_CloroPTE', 'Pred_CALCINOTE', 'Pred_CALCINPTE', 'Pred_ValNOxCuch',
       'Pred_ValTemSalMat01Ote', 'Pred_ValTemSalMat01Pte'])
    # ahora añadimos 4 columnas de zeros
    list_col = ["col0", "col1", "col2", "col3"]
    for c in list_col:
        mi_array_pred_df[c] = 0

    # ordeno los campo, dejando la coluna de prediccion a desnormalizar al final como estaba en df1_norm
    mi_array_pred_df = mi_array_pred_df[['col0', 'col1', 'col2', 'col3','Pred_KcalxKg',
       'Pred_Quality_CalLibre', 'Pred_TempAireSec', 'Pred_PresCaperuza', 'Pred_PresComp3',
       'Pred_PresComp4', 'Pred_SO3OTE', 'Pred_SO3PTE', 'Pred_SO3Clk', 'Pred_C3AClk', 'Pred_C3SClk', 'Pred_FSCClk',
       'Pred_CloroOTE', 'Pred_CloroPTE', 'Pred_CALCINOTE', 'Pred_CALCINPTE', 'Pred_ValNOxCuch',
       'Pred_ValTemSalMat01Ote', 'Pred_ValTemSalMat01Pte']]

    # ahora aplico la funcion de arriba que se aplica al dataframe df1 original
    prediccion = desnormalizar(mi_array_pred_df)  # pero ese dataframe incluye valores de los 4 ceros de relleno de las actuadoras
    #LOS ELIMINAMOS:
    prediccion = prediccion.drop(prediccion.columns[[0, 1, 2, 3]], axis=1)

    return prediccion


def predictall(list_values):
    list_features = [list_values + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    # print('\n','Input:',list_features)

    # Normalizacion de variables para ingresar al modelo
    # list_features_norm = normalizar(list_features_df).iloc[0]
    list_features_norm = normalizar(list_features)
    # print('FeatureNorm1',list_features_norm)

    # list_features_norm = normalizar([list_features])
    # print('FeatureNorm2',list_features_norm)
    # list_features_2 = list_features_norm.values.tolist()
    # print('Normalized:',list_features_2)
    #
    # # Se realiza la prediccion
    Result_pred = best_model_cargado.predict([list_features_norm[0][0:4]])
    # Result_pred = round(best_model_cargado.predict([list_features_norm[0][0:4]]),3)
    # Result_pred = round(best_model_cargado.predict())
    # # Result_pred = best_model_cargado.predict(list_features_2)
    # print('Result_pred:',Result_pred)
    Result_pred_desnorm = desnormalizar_pred(Result_pred)
    # print('Result:',Result_pred_desnorm,'\n')

    pred_list = Result_pred_desnorm.values.tolist()[0]
    type(pred_list)
    pred_list2 = [ '%.2f' % elem for elem in pred_list ]

    output = []
    for i in range(len(pred_list2)):
        output.append(float(pred_list2[i]))

    return output


# def calcula_variaciones(array, array_hist):
#     list_feat = array
#     hist_pi = array_hist
#
#     strings = []
#     for i in range(len(hist_pi)):
#         innerlist = []
#
#         change = (list_feat[i] - hist_pi[i]) / hist_pi[i] * 100
#         variation = round(change, 1)
#
#         if variation > 0:
#             string = '▲ ' + str(variation) + '%'
#         else:
#             string = '▼ ' + str(variation) + '%'
#
#         strings.append(string)
#
#     return strings

### nueva funcion para carga termica ------------------------
import math
pi = math.pi

def CargaTermica(Tag1,Tag2,OvenDiameter,RefractoryThickness,UnitOD,UnitRT):

    if UnitOD == 'centimeters':
        OvenDiameter = OvenDiameter/100
    if UnitRT == 'centimeters':
        RefractoryThickness = RefractoryThickness/100

    Factor = math.pow((OvenDiameter - 2*RefractoryThickness),2)
    CargaTermica = (Tag1 * Tag2) / ((pi/4) * Factor * 1000)

    return CargaTermica

### Temp: valores referencia --------------------------------------------------

lista_referencia = ['KcalxKg', 'Quality_CalLibre', 'TempAireSec', 'PresCaperuza', 'PresComp3',  #'HeatPetCoke','HeatRefuseFuel','VTIid06','VTIid07',
                       'PresComp4', 'SO3OTE', 'SO3PTE', 'SO3Clk', 'C3AClk', 'C3SClk', 'FSCClk',
                       'CloroOTE', 'CloroPTE', 'CALCINOTE', 'CALCINPTE', 'ValNOxCuch',
                       'ValTemSalMat01Ote', 'ValTemSalMat01Pte']

lista_referencia2=['OvenDiameter','RefractoryThickness'] # este es nuevooooooooo

lr_min = [ 122000,27805,709,703,811,0.4,800,-5,468,468,2,2,3,4,56,99,0,0,88,89,68,799,814]

lr_max = [ 210000, 117000, 731, 731, 1185, 1.5, 1200, 5, 802, 805, 6,6,7,8,65,102,1.5,1.5,95,93,1030,881,880]

# out = ['a','b','c','d','e','f']

# hist_pi = [909,1.3,900,1.5,577,482,3.8,4.4,3.5,5.2,58,100,0.28, 0.71,91.42,91.98,559,823,868]
# value_dict = [6500,4120,250,176]

users = {'user': ['admin','dev','operator','alopez','fvaldez','klekston','afuente','jmoncada','ralvarez','cmedina'],
        'pwd': ['soyadmin','c0gnitive_Team','pvO7Qg8g','c0gnitive_Team','c0gnitive_Team','OF-%Zh,b','f9DS_\V7','eb.w"FZ9','l9NjJEV7','B/5FPuM$']}
users_db = DataFrame(users,columns= ['user', 'pwd'])



# --- Flask init ---------------------------------------------------------------
app = Flask(__name__)
Mobility(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
"""
features_input= ['HeatPetCoke',' HeatRefuseFuel','HeatResiduoCarb','HeatFuelOiL','TPHClk','VelHorno','VTIid06','VTIid07','PresCaperuza','PresComp3','PresComp4']

"""
## ----- Route setup
@app.route('/')
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return index()

@app.route('/login', methods=['POST'])
def do_admin_login():

    pwd = users_db.loc[users_db.user == request.form['username'], 'pwd']

    if (pwd == request.form['password']).any():
        session['logged_in'] = True
    else:
        flash('wrong password!')
    return home()

    # if request.form['password'] == 'password' and request.form['username'] == 'admin':
    #     session['logged_in'] = True
    # else:
    #     flash('wrong password!')
    # return home()

@app.route('/dash')
# @mobile_template('{mobile/}index.html')
@mobile_template('index.html')
def index(template):

    import json
    response1 = open("/Users/anamendoza/Desktop/Cemex Ventures/Intelligence Klinn/flask_19nov/flask-app/inputs.txt","r")
    if response1.mode == "r":
        contents = response1.read()
        value_dict = json.loads(contents)

    response2 =  open("/Users/anamendoza/Desktop/Cemex Ventures/Intelligence Klinn/flask_19nov/flask-app/monitor.txt","r")
    if response2.mode == "r":
        contents2 = response2.read()
        out = json.loads(contents2)
    tag_list = out['tags']

    pred_list2 = predictall(value_dict)

    return render_template(template, random_list = value_dict, names = lista_referencia,
                           tags = tag_list, Result_pred=pred_list2,
                           ref = out, min_ref = lr_min, max_ref = lr_max , min_max = stats)


@app.route('/result', methods=['POST'])
# @mobile_template('{mobile/}index.html')
@mobile_template('index.html')
def index_post(template):

    import json
    response1 = open("/Users/anamendoza/Desktop/Cemex Ventures/Intelligence Klinn/flask_19nov/flask-app/inputs.txt","r")
    if response1.mode == "r":
        contents = response1.read()
        value_dict = json.loads(contents)

    # Recoger variables de input
    HeatPetCoke = request.form['HeatPetCoke']
    HeatRefuseFuel = request.form['HeatRefuseFuel']
    VTIid06 = request.form['VTIid06']
    VTIid07 = request.form['VTIid07']
    listof_features = [HeatPetCoke,HeatRefuseFuel,VTIid06,VTIid07]

    response2 = open("/Users/anamendoza/Desktop/Cemex Ventures/Intelligence Klinn/flask_19nov/flask-app/monitor.txt","r")
    if response2.mode == "r":
        contents2 = response2.read()
        out = json.loads(contents2)
    tag_list = out['tags']

    pred_list2 = predictall(listof_features)


    return render_template(template, random_list = value_dict, names = lista_referencia,
                           tags = tag_list, Result_pred=pred_list2,
                           ref = out, min_ref = lr_min, max_ref = lr_max)


@app.route('/help')
# @mobile_template('{mobile/}help.html')
@mobile_template('help.html')
def help(template):
    return render_template(template)

# ---- NUEVO TAB Carga Termica ----------------

@app.route('/NewTab')
@mobile_template('SecondIndex.html')
def NewTab(template):

    import json
    response1 = open("/Users/anamendoza/Desktop/Cemex Ventures/Intelligence Klinn/flask_19nov/flask-app/inputs.txt","r")
    if response1.mode == "r":
        contents = response1.read()
        value_dict = json.loads(contents)

    response2 =  open("/Users/anamendoza/Desktop/Cemex Ventures/Intelligence Klinn/flask_19nov/flask-app/monitor.txt","r")
    if response2.mode == "r":
        contents2 = response2.read()
        out = json.loads(contents2)
    tag_list = out['tags']

    return render_template(template, Result_pred='-')


@app.route('/NewTabResult', methods=['POST'])
@mobile_template('SecondIndex.html')
def NewTab_post(template):

        import json
        response1 = open("/Users/anamendoza/Desktop/Cemex Ventures/Intelligence Klinn/flask_19nov/flask-app/inputs.txt","r")
        if response1.mode == "r":
            contents = response1.read()
            value_dict = json.loads(contents)

        # captura inputs: valores y unidades de medición
        OvenDiameter = float(request.form['OvenDiameter'])
        RefractoryThickness = float(request.form['RefractoryThickness'])
        Tag1 = 100
        Tag2 = 100
        UnitOD = request.form['unitOD']
        UnitRT = request.form['unitRT']

        response2 = open("/Users/anamendoza/Desktop/Cemex Ventures/Intelligence Klinn/flask_19nov/flask-app/monitor.txt","r")
        if response2.mode == "r":
            contents2 = response2.read()
            out = json.loads(contents2)
        tag_list = out['tags']

        # calcula la carga térmica
        pred = CargaTermica(Tag1,Tag2,OvenDiameter,RefractoryThickness,UnitOD,UnitRT)

        return render_template(template, Result_pred="%.2f" % pred)

# --- Flask run ------------------------------------------------------------
#import time
#time.sleep(1.5)
if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    #app.run(debug=True)
    # app.run(host='0.0.0.0', port='50512', debug=True)
    # app.run(host='0.0.0.0', debug=True) #for flask
    # app.run(host='10.26.192.26', port='50512', debug=False) #serverEnergia
    app.run(host='0.0.0.0', port='50512', debug=False) #serverGA
