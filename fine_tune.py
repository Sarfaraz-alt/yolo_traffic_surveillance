from ultralytics import YOLO
def fine_tune(model_path=None,data_path=None,**Kwargs):
    try:
        if model_path == None:
            raise FileNotFoundError("No Model Path Given")
        if data_path == None:
            raise FileNotFoundError("No Data Provided")

        model = YOLO(model_path)
        results = model.train(data = data_path,**Kwargs)
        return results
    except Exception as e:
        print(e)

if __name__ == '__main__':
    fine_tune()