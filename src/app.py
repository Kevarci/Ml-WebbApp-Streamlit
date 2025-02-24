import streamlit as st
import numpy as np 
import pandas as pd 
import matplotlib as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 

def main():
    st.title('Solucion del porject Ml WebbApp with StreamLit')
    st.header('Esto es un encabezado')
    st.subheader('Esto es un subencabezado')
    st.text('esto es un texto')

    nombre = 'kevin'

    st.text(f'Mi nombre es {nombre} el estudiante')

    st.success('Este es mi mensaje de aprobado con exito')
    st.warning('no se visualiza el project ')
    st.info('Tarea rechazada, vuelva a enviar')

if __name__=='__main__':
    main()

#iniciar con el modulo

np.random.seed(42)

X = np.random.rand(100, 1)*10
y = 3* X +8 + np.random.randn(100,1)*2

#Separar el conjuunto de datos train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#generar el modelo
model= LinearRegression()
model.fit(X_train, y_train)

#prediccion
ypred = model.predict(X_test)
mse= mean_squared_error(y_test, ypred)

#interfaz 
st.title('Mi primer regresion lineal en web')
st.write('Este es un  modelo de ejemplo para entrenar el proyecto')

#usar un  select box
opcion = st.selectbox('selecionar el tipo para entregar el project', ['Dispersion', 'Linea de Regresion'])

#checkbox para mostrar
if st.checkbox('Mostra coeficientes del modelo'):
    st.write(f'Coeficiente: {model.coef_[0][0]:.2f}')
    st.write(f'Coeficiente: {model.intercept_[0]:.2f}')
    st.write(f'Error medio cuadratico: {mse}:.2f')

#slider
    data_range = st.slider("Seleccione el rago que quiere evaluar",0,100,(10,90) )
    x_display = X_test[data_range[0]:data_range[1]]
    y_display = y_test[data_range[0]:data_range[1]]
    y_pred_display = ypred[data_range[0]:data_range[1]]

#visualizacion
    fig, ax = plt.subplots()
    if opcion == "Dispersion":
        ax.scatter(x_display, y_display)
        plt.title("Grafico de Dispersion")
    else:
        ax.plot(x_display, y_pred_display)
        plt.title("Grafico de Linea")
    st.pyplot(fig)
    opcion_2 = st.multiselect('Informacion Adicional', ['Informe General', 'Prediccion'])
    if 'Informe General' in opcion_2:
        st.write('Aca deberian estar los datos!')
        st.write(pd.DataFrame({'X Test': X_test.flatten(), 'y Test': y_test.flatten(), 'y Predict':ypred.flatten()}))
    if 'Prediccion' in opcion_2:
        st.write('Aca tambien deberian haber datos!')
        for i in range(len(x_display)):
            st.write(f'x = {x_display[i][0]:.2f} prediccion y = {y_pred_display[i][0]:.2f}')
    #recibir informacion de usuario
    st.header('Ingrese su valor de X')
    x_ingresado = st.number_input('Numero a predecir')
    if st.button('Calcular'):
        prediccion_x = model.predict([[x_ingresado]])[0][0]
        st.write(f'La preddiccion para {x_ingresado} es {prediccion_x}')
if __name__ == '__main__':
    main()