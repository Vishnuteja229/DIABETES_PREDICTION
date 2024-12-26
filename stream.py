import numpy as np
import pickle
import streamlit  as st

loaded_model=pickle.load(open("C:/Users/vishn.000/PYTHON CODES/MACHINE_LEARNING/MACHINE_LEARNING/DIABETES_PREDICTION/trained.sav",'rb'))

def diab_predict(data):
    np_arr=np.asarray(data)
    reshaped=np_arr.reshape(1,-1)
    
    prediction=loaded_model.predict(reshaped)
    if(prediction==0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    
def main():
    st.title('Diabetes prediction')
    Pregancies= st.text_input('no.of pregnancies')
    Glucose=st.text_input('Glucose') 
    BloodPressure=st.text_input('BloodPressure')
    SkinThickness=st.text_input('SkinThickness')
    Insulin=st.text_input('Insulin')
    BMI=st.text_input('BMI')
    DiabetesPedigreeFunction=st.text_input('DPF')
    Age=st.text_input('Age')
    
    diagnosis=''
    if st.button('check results'):
        diagnosis=diab_predict([Pregancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    st.success(diagnosis)    
    
if __name__=='__main__' :
    main()   
    