import os

from ultralytics import YOLO,solutions
import  streamlit as st
class HyperParameters:
    imgsz = 640
    model_ls = os.listdir('./models_dir')
    model_ls = [i.removesuffix('.pt') for i in model_ls]
    task_ls = ['Detection','Count']
    conf = 0.5


class Models:
    @st.cache_resource
    @staticmethod
    def load_model(model_name,conf=HyperParameters.conf):
        if model_name == 'All':
            ls = []
            for i in HyperParameters.model_ls:
                if i != 'All':
                    ls.append(Models.load_model(i))

            return ls
        else:
            path = f'./models_dir/{model_name}.pt'
            return  YOLO(path)

    @st.cache_resource
    @staticmethod
    def load_counter(model_path,region,tracker,**Kwargs):
        return  solutions.ObjectCounter(
            show=False,
            model=model_path,
            region=region,
            tracker=tracker,
            **Kwargs
        )