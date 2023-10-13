import datetime
import json

def get_current_time():
    now = datetime.datetime.now()
    str_now = str(now).split('.')[0].replace(' ', '_').replace(':', '_')
    return str_now

def load_imagenet_classes(json_path):
    with open(json_path) as f:
        str_classes = json.load(f)
        f.close()
    classes = dict(map(lambda x: (int(x), str_classes[x]), str_classes.keys() ))
    return classes